from typing import Optional

class NANDOp(object):
    """
    NAND Operation computes NAND with inputs a and b
    """
    def __init__(self, a, b) -> None:
        """
        initialize NAND Operation with inputs a and b
        Arguments:
            a: NANDOp or InputOp = first input to NAND
            b: NANDOp or InputOp = second input to NAND
        Returns:
            None
        """
        self.a: NANDOp | InputOp = a
        self.b: NANDOp | InputOp = b
        self.state: Optional[bool] = None
    
    def __call__(self) -> bool:
        """
        result of NAND evaluation of instance's a and b
        Returns:
            bool
        """
        if self.state is None:
            self.state = not (self.a() and self.b())
        return self.state
    
    def reset_state(self) -> None:
        """
        resets state to None
        Returns:
            None
        """
        self.state = None
        self.a.reset_state()
        self.b.reset_state()

class InputOp(object):
    """
    Input Operation stores input value and passes it to NAND Operation
    """
    def __init__(self, default_val: bool = 0, name: str = "") -> None:
        """
        initializes Input Operation with null input
        Arguments:
            default_val: bool = default input value
            name: str = input name
        """
        self.name: str = name
        self.val: bool = default_val
    
    def __call__(self) -> bool:
        """
        input value
        Returns:
            bool
        """
        return self.val
    
    def set_val(self, val: bool) -> None:
        """
        sets instance's input value
        Arguments:
            val: bool = input value
        Returns:
            None
        """
        self.val = val
    
    def reset_state(self) -> None:
        """
        catches state reset propogations
        Returns:
            None
        """
        pass