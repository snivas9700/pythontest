from sys import stdout

class Scribe(object):
    """
        Object to control what gets printed to the console based on a tiered system. The Scribe provides messages with a prepended, color coded label that categories each message outputed to a level. The Scribe also has a minimum threshold that must be below the level of message before it can be sent. By changing the threshold the Scribe will print less information or more information, depending on if you decrease or increase it respectively. 
    """
    def __init__(self, level, colors=True):
        """
            Defaults are stdout as the output location and colors to be true. Level must be provided on creation, if you don't care you don't need to use this to write your content.
        """
        self.set_thresh(level)
        self.scroll = stdout
        self.colors = colors
        
    def set_thresh(self, level):
        """
        Sets the lowest level of messages to output. Each message is assigned a level went sent to the scribe to be 
        scribed; if the level's value (see _get_level) is higher than the threshold set the message is sent. The lower 
        the threshold, the more detailed reporting. The threshold is set by providing either the corresponding level's 
        name or the numerical value of the letter, for ease of use. 
        """
        self.level_threshold = self._get_level(level)[1]
        
    def set_output(self, f=None):
        """
            Sets the output file to have the Scribe scribe to. If nothing is provided or a non file object is provided, 
            the output is reset to default stdout.
        """
        if f is None or not isinstance(f, file):
            f = stdout
        self.scroll = f
        
    def _get_level(self, level):
        """
            Takes an input and turns it into a level tuple. Levels are represented by three values, their prepend label 
            their numerical ranking, and their color code. The prepend is placed in front of every message that is 
            scribed at that level. The color code is used when colors is true (default) to differentate them quickly. 
            The numerical value is used to decide if a message should be sent, if the numerical value of a level is 
            lower than the scribe's threshold level, it's not sent. Allows for filtering of the lower importance 
            messages. The levels are listed in numerical order below.
            ALL < TRACE < DEBUG < INFO < WARN < ERROR < FATAL < SHOUT < OFF
        """
        if type(level) is not str:
            level = repr(level)
        level = level.lower()
        if level in ['0', 'all']:
            return ('  ALL', 0, '\033[0m')
        elif level in ['1', 'trace']:
            return ('TRACE', 1, '\033[37m')
        elif level in ['2', 'debug']:
            return ('DEBUG', 2, '\033[36m')
        elif level in ['3', 'info', 'information']:
            return (' INFO', 3, '\033[32m')
        elif level in ['4', 'warn', 'warning']:
            return (' WARN', 4, '\033[33m')
        elif level in ['5', 'error']:
            return ('ERROR', 5, '\033[31m')
        elif level in ['6', 'fatal']:
            return ('FATAL', 6, '\033[31;1m')
        elif level in ['7', 'notis']:
            return ("NOTIS", 7, '\033[32;1m')
        else: # level in ['8', 'off', 'disable', 'quiet', 'hush']:
            return ('  OFF', 8, '\033[0m')
    
    def scribe(self, messages, level='debug', newline='\n', prompting='first'):
        """
            Write to the configured output file (default is stdout). It can take a string or a list of strings as an 
            input, for printing datasets or long strings. Defaults the level to debug if no level is provided. Allows 
            overriding the default behavior of a newline at the end of each string in the messages with whatever you 
            provide instead. Prepends the message with the level variable's prepend and color. Does not print when the
            threshold of the Scribe is greater than the level. 
        """
        if type(messages) is not list:
            messages = [messages]
            
        prompting_dict = {'none':0, 'first':1, 'all':len(messages)}
        if prompting in prompting_dict: 
            prompting = prompting_dict[prompting]
        else:
            prompting = 1
            self.scribe("Invalid prompting input received, defaulting to first.", 'error')
        
        for msg in messages:
            level_info = self._get_level(level)
            if level_info[1] >= self.level_threshold: # Printing this level of message.
                if type(msg) is not str:
                    msg = str(msg)
                if prompting:
                    if self.colors:
                        prompt = level_info[2] + level_info[0] + ":\033[0m  "
                    else:
                        prompt = level_info[0] + ':  '
                    prompting -= 1
                else: 
                    prompt = '        '
                output = prompt + msg + newline
                self.scroll.write(output)
                
if __name__ == "__main__":
    # Does nothing, just prints a sample of every level's look.
    s = Scribe('all')
    for x in range(0,9):
        s.scribe('Some kind of message.', x)