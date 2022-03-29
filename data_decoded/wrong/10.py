Module(If(Compare(Name, Gt, Constant), Then(Expr(Call(Name, BinOp(Name, Mult, Constant)))), Else))
Module(If(Compare(Name, Eq, Constant), Then(Expr(Call(Name, Constant))), Else))