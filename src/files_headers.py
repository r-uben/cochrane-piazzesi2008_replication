    

class FilesHeaders:

    def y_header(n) -> str: 
        if len(str(n)) == 1: 
            output = "SVENY0" + str(n)
        else: 
            output = "SVENY" + str(n)
        return output
    
    def p_header(n) -> str: return "p"+str(n)

    def f_header(n) -> str: return "f"+str(n)

    def r_header(n) -> str: return "r"+str(n)

    def rx_header(n) -> str: return "rx"+str(n)