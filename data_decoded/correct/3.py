Module(Assign(Name, Call(Name, Call(Name, Constant))), If(UnaryOp(Not, Compare(Name, LtE, Constant)), Then(Expr(Call(Name, BinOp(Name, Mult, Constant)))), Else(Expr(Call(Name, Constant)))))
Module(Assign(Name, Call(Name, Call(Name, Constant))), If(BoolOp(Or, Compare(Name, Eq, Constant), Compare(Constant, LtE, Name)), Then(Expr(Call(Name, BinOp(Constant, Add, Name)))), Else(Expr(Call(Name, Constant)))))