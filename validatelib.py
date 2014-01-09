import os, sys, re, time, subprocess, urllib.request, stat
from multiprocessing import Process, Value, Array, Pipe

class Log:
    def print(text=""):
        print(text)
        sys.stdout.flush()
    def printStdOut(out, maxoutputchars):
        if len(out) > maxoutputchars:
            out = out[:maxoutputchars] + "..."
        Log.print(out)

class ErrorMessage(Exception):
    def __init__(self, message, errorcode=1):
        self.message = message
        self.errorcode = errorcode
    def __str__(self):
        return repr(self.message) + " [" + repr(self.errorcode) + "]"
    def show(self):
        Log.print("[ERROR]-----------------------------")
        Log.print(self.message)
        Log.print("------------------------------------")

class ExecutionInfo:   
    maxThreadCount = -1
    runtime = 0
    stdout=None
    stderr=None
    returncode=None
    
    maximalthreadcount=Value('i',0)
    
    maxoutputchars = 500
    
    def __init__(self,description,executeable,parameters,output,expectedThreadCount=-1,expectedRuntime=-1,isspecial='no'):
        self.isspecial = isspecial
        self.description = description
        self.executeable = FileInfo(executeable)
        self.output = output
        self.expectedThreadCount = expectedThreadCount
        self.expectedRuntime = expectedRuntime
        self.parameters = parameters
        for i in range(0, len(self.parameters)):
            self.parameters[i] = str(self.parameters[i])
    
    def clearPerformanceFile(self):
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            open(sys.argv[1], 'w').close()
            
    def measure(self):
        if len(sys.argv) < 2:
            Log.print("No filename for performance evaluation results given!")
            return 10
        evaluationFile = sys.argv[1]
        result = self.run()
        if not result == 0:
            return 10 + result
        evaluationText = "{};{:.10f}".format(self.maxThreadCount,self.runtime)
        Log.print('writing "' + evaluationText + '" to file ' + evaluationFile)
        with open(evaluationFile, "w") as f:
            f.write(evaluationText)
        return 0
        
    def runall(executionInfos):
        result = 0
        for info in executionInfos:
            result += info.run()
            Log.print()
        return result
    
    def run(self):
        try:
            self.__run()
        except ErrorMessage as e:
            e.show()
            return e.errorcode
        return 0

    def __run(self):
        Log.print('Test description: ' + self.description)
        self.clearPerformanceFile()
        if self.isspecial == 'scala':
            self.executeable.ensurefileexists('.class')
        else:
            self.executeable.ensurefileexists()
        self.output.removeifexistent()
        
        try:
            if self.isspecial == 'java':
                Log.print('Running:          java -jar ' + str(self.executeable) + ' ' + ' '.join(self.parameters))
                self.__execute('java', ['-jar', self.executeable.name] + self.parameters)
            elif self.isspecial == 'scala':
                Log.print('Running:          scala -classpath . ' + str(self.executeable) + ' ' + ' '.join(self.parameters))
                self.__execute('scala', ['-classpath', '.', self.executeable.name] + self.parameters)
            else:
                Log.print('Running:          ' + str(self.executeable) + ' ' + ' '.join(self.parameters))
                self.__execute(self.executeable.name, self.parameters)  
        finally:
            Log.printStdOut(self.stdout, self.maxoutputchars)
            Log.printStdOut(self.stderr, self.maxoutputchars)
            Log.print("Exit code:        {}".format(self.returncode))

        if self.expectedThreadCount == -1:
            if self.runtime < 0.11:
                Log.print("Max thread count: {} [Estimated!]".format(self.maxThreadCount))
            else:
                Log.print("Max thread count: {}".format(self.maxThreadCount))
        else:
            threadcountquality = "Ok"
            if self.maxThreadCount > self.expectedThreadCount + 1:
                threadcountquality = "Too Many!"
            if self.maxThreadCount < self.expectedThreadCount:
                threadcountquality = "Too Few!"
            if self.runtime < 0.11:
                threadcountquality = "Estimated!"
            Log.print("Max thread count: {} [{}]".format(self.maxThreadCount, threadcountquality))
     
        if self.expectedRuntime == -1:
            Log.print("Runtime:          {:.2f}".format(self.runtime))
        else:
            runtimequality = "Ok"
            if self.runtime < 0.9 * self.expectedRuntime:
                runtimequality = "Too Fast!"
            if self.runtime > 1.1 * self.expectedRuntime:
                runtimequality = "Took to Long!"
            Log.print("Runtime:          {:.2f} [{}]".format(self.runtime, runtimequality))
        
        self.output.ensurefileexists()
        self.output.checkfile()
    
    def __execute(self, command, parameters):
        self.maximalthreadcount.value = 0
        pipeRecv, pipeSend = Pipe(duplex=False)
        pollingthread = Process(target=self.__pollingthreadbody, args=(pipeRecv,))
        pollingthread.start()
        start = time.time()
        self.returncode = None
        proc = subprocess.Popen([command] + parameters, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        pipeSend.send([proc.pid])
        self.stdout, self.stderr = proc.communicate()
        end = time.time()
        pollingthread.terminate()
        self.returncode = proc.returncode
        if self.stdout == None:
            self.stdout = "" 
        else:
            self.stdout=self.stdout.decode("utf-8")
        if self.stderr == None:
            self.stderr = "" 
        else:
            self.stderr=self.stderr.decode("utf-8")
        self.maxThreadCount = self.maximalthreadcount.value
        self.runtime = end - start
        
    def __pollingthreadbody(self,pipeRecv):
        #Log.print("Waiting for PID")
        processid=pipeRecv.recv()[0]
        #Log.print("Got PID "+str(processid))
        while True:
            #Log.print("Poll")
            try:
                with open('/proc/{}/status'.format(processid)) as f:
                    for threadline in f:
                        if threadline.startswith('Threads:'):
                            currentthreadcount = int(threadline.split('\t')[1])
                            break
            except IOError: # could not open status file
                time.sleep(0.05)
                continue
            if currentthreadcount is not None and currentthreadcount > self.maximalthreadcount.value:
                #Log.print("Found %u threads in application"%currentthreadcount)
                self.maximalthreadcount.value = currentthreadcount
            time.sleep(0.05)
        
class FileInfo:
    name = "output.txt"        
    
    def __init__(self,name,ideal=None,idealsize=-1):
        self.name = name        
        self.ideal = None
        self.idealsize = idealsize
        if ideal is not None:
            self.ideal = self.newFileInfo(ideal)
    
    def newFileInfo(self, name):
        return FileInfo(name)

    def __str__(self):
        return self.name
        
    # check if file exists
    def ensurefileexists(self,extension=''):
        name = self.name+extension
        if not os.path.exists(name) or not os.path.isfile(name):
            raise ErrorMessage("File not found: " + os.path.realpath(name), 1)
        if os.path.getsize(name) == 0:
            raise ErrorMessage("File is empty: " + os.path.realpath(name), 1)
        
    def removeifexistent(self):
        if os.path.exists(self.name) and os.path.isfile(self.name):
            os.remove(self.name)
            
    # download text file via URL
    def loadFromUrl(self, url):
        if os.path.exists(file) and os.path.isfile(file) and not os.path.getsize(file) == 0:
            return
        fileHandle = urllib.request.urlopen(url)
        filecontent = self.readContent(fileHandle)
        fileHandle.close()    
        if os.path.exists(file):
            os.chmod(file, stat.S_IWOTH | stat.S_IWGRP | stat.S_IWUSR)
        self.writeContent(file, filecontent)
        os.chmod(file, stat.S_IROTH | stat.S_IRGRP | stat.S_IRUSR)     

    def readContent(self, fileHandle):
        return fileHandle.read()

    def writefile(self, filename, filecontent):                  
        with open(filename, self.writeModifiers()) as f:
            f.write(filecontent)  

    def writeModifiers(self):
        return "wb"              
    
    def readfile(self, filename):        
        # try open file
        try:
            with open(filename, self.readModifiers()) as f:
                return f.read()
        except IOError as e:
            raise ErrorMessage("Could not open file: " + name, 2)    

    def readModifiers(self):
        return "rb"              

    # check if file content matches pattern
    def checkfile(self):
        content = self.readfile(self.name)
        self.checkcontent(content)
        if self.ideal is not None:
            self.checkideal(content)
        if self.idealsize > -1 and len(content) != self.idealsize:
            raise ErrorMessage("Result file does not have the expected file size of " + str(self.idealsize) + ".", 3)
    
    def checkcontent(self, content):
        if len(content) == 0:
            raise ErrorMessage("Result file was empty.", 3)

    def checkideal(self, content):
        idealcontent = self.readfile(self.ideal.name)
        if not self.compare(content, idealcontent):
            self.raiseContentProblem(content, idealcontent)

    def compare(self, content, idealcontent):
        return content == idealcontent

    def raiseContentProblem(self, content, idealcontent, add=""):
        if add != "":
            add = "\n\n" + add
        raise ErrorMessage("Result file does not have the expected content:\n\n" + self.name + "\n" + self.forMessage(content) + "\n\n" + self.ideal.name + "\n" + self.forMessage(idealcontent) + add, 3)    

    def forMessage(self, content):
        return "Binary file of size " + str(len(content)) + "."

class TextFileInfo(FileInfo):
    format = "^.*?$"
    maxoutputchars = 500

    def __init__(self,name,format="^.*?$",ideal=None):
        FileInfo.__init__(self, name, ideal)
        self.format = format

    def newFileInfo(self,name):
        return TextFileInfo(name)

    def readContent(self, fileHandle):
        mybytes = fileHandle.read()
        return mybytes.decode("utf8")  

    def writeModifiers(self):
        return "w"

    @staticmethod
    def readModifiers():
        return "r"                            
    
    # check if text is in the correct format
    @staticmethod
    def matches(text, format):
        match = re.search(format, text, re.DOTALL)
        if match is None:
            return False
        return match.group(0) != ""

    # check if file content matches pattern
    def checkcontent(self, content):
        if not TextFileInfo.matches(content, self.format):
            if len(content) > self.maxoutputchars:
                content = content[:self.maxoutputchars] + "..."
            raise ErrorMessage("Result file does not match regular expression:\n" + self.format + "\n\n" + self.name + "\n" + content, 3)

    def prepareTexts(self,text):
        try:
            with open(self.ideal.name) as fIdeal:
                text = text.replace("\r", "")
                if text[len(text)-1] == '\n':
                    text = text[:len(text)-1]
                textIdeal = fIdeal.read().replace("\r", "")
                if textIdeal[len(textIdeal)-1] == '\n':
                    textIdeal = textIdeal[:len(textIdeal)-1]
                return [text, textIdeal]                
        except IOError as e:
            raise ErrorMessage("Could not open file: " + self.ideal.name, 2)

    def forMessage(self, content):
        if len(content) > self.maxoutputchars:
            content = content[:self.maxoutputchars] + "..."
        return content

    def readfile(self, name):
        text = FileInfo.readfile(self,name).replace("\r", "")
        if text[len(text)-1] == '\n':
            text = text[:len(text)-1]
        return text

class NumberListFileInfo(TextFileInfo):
    allowedDifference = 0.000001

    def __init__(self,name,format="^.*?$",ideal=None,allowedDifference=0.000001):
        TextFileInfo.__init__(self, name, format, ideal)
        self.allowedDifference = allowedDifference

    def newFileInfo(self,name):
        return NumberListFileInfo(name)

    def compare(self, content, idealcontent):
        textAsLines = content.split('\n')
        textIdealAsLines = idealcontent.split('\n')

        for i in range(0, len(textAsLines)):
            if abs(float(textAsLines[i]) - float(textIdealAsLines[i])) > self.allowedDifference:
                if abs(float(textAsLines[i]) - float(textIdealAsLines[i])) > self.allowedDifference * float(textIdealAsLines[i]):
                    self.raiseContentProblem(content, idealcontent, "Line " + str(i+1) + ": " + str(float(textAsLines[i])) + " != " + str(float(textIdealAsLines[i])) + " +/-" + str(self.allowedDifference * float(textIdealAsLines[i])))
        return True