import sys
from validatelib import ExecutionInfo, TextFileInfo

if __name__ == '__main__':

    print('The last three runs provide a thread count as the first parameter, since the old assigment description used them.')
    print('You will pass this task if your program works correctly for either the first three or the last three runs.')
    print()

    res = ExecutionInfo.runall([
        ExecutionInfo('assigment example', './parsum', ['1',  '1000000000'], TextFileInfo('output.txt', '^500000000500000000$')),
        ExecutionInfo('weird count', './parsum', ['1',  '911'], TextFileInfo('output.txt', '^415416$')),
        ExecutionInfo('starting index > 1', './parsum', ['912', '1000000000'], TextFileInfo('output.txt', '^500000000499584584$')),

        ExecutionInfo('assigment example [with obsolete thread count]', './parsum', ['8', '1',  '1000000000'], TextFileInfo('output.txt', '^500000000500000000$')),
        ExecutionInfo('weird count [with obsolete thread count]', './parsum', ['8', '1',  '911'], TextFileInfo('output.txt', '^415416$')),
        ExecutionInfo('starting index > 1 [with obsolete thread count]', './parsum', ['8', '912',  '1000000000'], TextFileInfo('output.txt', '^500000000499584584$')),
    ])

    sys.exit(res)