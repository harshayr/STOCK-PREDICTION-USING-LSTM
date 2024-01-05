import sys 

def error_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"Error occur in python script [{0}] lineno [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, error
    )
    return error_msg

class CustomException(Exception):

    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error_msg = error
        self.error_detail = error_detail

    def raise_error(self):
        return error_details(self.error_msg,self.error_detail)
    
    
    



