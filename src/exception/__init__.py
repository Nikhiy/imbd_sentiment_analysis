import sys
import logging
import traceback


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Function Name :   error_message_detail
    Description   :   Extracts concise error information (file name, line number,
                      and exception message) without dumping large objects like
                      MongoDB data, DataFrames, or NumPy arrays.

    :param error: The original exception object
    :param error_detail: The sys module to extract traceback info
    :return: A clean, formatted error message string
    """

    # exc_info() returns (type, value, traceback)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Get the exact line number that caused the exception
    line_number = exc_tb.tb_lineno

    # Extract ONLY the exception message (not full objects)
    # This prevents printing Mongo data / DataFrames
    error_msg = error.args[0] if error.args else type(error).__name__

    # Final clean error message
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}] | "
        f"Error message: {error_msg}"
    )

    return error_message


class MyException(Exception):
    """
    Class Name  :   MyException
    Description :   Custom exception class used across the project
                    to standardize error messages and avoid noisy logs.
    """

    def __init__(self, error: Exception, error_detail: sys):
        """
        Initializes the custom exception with a clean error message.

        :param error: Original exception object
        :param error_detail: sys module for traceback extraction
        """

        # Generate concise error message
        self.error_message = error_message_detail(error, error_detail)

        # Call base Exception class
        super().__init__(self.error_message)

    def __str__(self) -> str:
        """
        Returns the string representation of the exception.
        """
        return self.error_message