def submatch(text, string):
    '''
    locate the first occurance of a string in a text
    '''
    
    i = 0
    for j, letter in enumerate(text):
        if letter == string[i]:
            i += 1

            if i == len(string):
                return j - len(string) + 1, j + 1

        else:
            i = 0

    return -1, -1

def cleanText(text):
    '''
    Remove spaces from start and end of text
    '''
    
    if text[0] == ' ':
        text = text[1:]
                
    if text[-1] == ' ':
        text = text[:-1]

    return text

def formatCommand(command):
    '''
    Split the command into text and parameters
    '''
    
    parts = []
    p, i = 0, 0

    while i < len(command):
        if not(command[i].isalpha() or command[i] == ' '):
            parts.append(cleanText(command[p:i]))
            parts.append(command[i:i+3])
            
            i += 2
            p = i+1

        i += 1

    if command[i-1].isalpha():
        parts.append(cleanText(command[p:i]))

    return parts

class Handler:
    '''
    Class to read command mapping file and redirect user input to appropriate function
    '''

    def __init__(self, mappings):
        '''
        Collect command/function pairs
        '''
        
        with open(mappings, 'r') as f:
            data = f.read().split('\n')

        self.commands = []
        self.functions = []

        for i, line in enumerate(data[:-1]):
            if data[i-1] == '{':
                command = line.split(',')

            if data[i+1] == '}':
                function = line

                for c in command:
                    c = formatCommand(c)
                    
                    self.commands.append(c)
                    self.functions.append(function)

    def match(self, command):
        '''
        match a user input to its appropriate function with correct parameters
        '''
        
        matches = []
        
        for i, c in enumerate(self.commands):
            params = {}
            param = False
            tmpcmd = command
            
            for section in c:
                if section[0].isalpha():
                    a, b = submatch(tmpcmd, section)
                    
                    if a < 0:
                        break

                    if param:
                        params[param] = cleanText(tmpcmd[:a])

                    tmpcmd = tmpcmd[b:]
                    
                else:
                    param = section

                if a < 0:
                    break

            if a >= 0:
                if tmpcmd != '':
                    params[section] = cleanText(tmpcmd)
        
                if params == {}:
                    params['_'] = '_'

                function = self.functions[i]

                matches.append([function, params])
                    
        if matches:
            matches = sorted(matches, key=lambda x: -len(x[1]))
            
            function = matches[0][0]
            
            for param in matches[0][1]:
                function = function.replace(param, '"' + matches[0][1][param] + '"')

            return function

        return ''
                    
                    
                
