Module(Assign(Name, Call(Name, Call(Name))), If(Compare(Name, Gt, Constant), Then(Expr(Call(Name, BinOp(Name, Mult, Constant)))), Else(Expr(Call(Name, Constant)))))
Module(Assign(Name, Constant), If(Compare(Name, Gt, Constant), Then(Expr(Call(Name, Constant))), Else(Expr(Call(Name, Constant)))))