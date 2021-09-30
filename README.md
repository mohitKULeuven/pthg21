# pthg21

## Tias' notes on a bias for initial constraint learner
We can start with learning constraints at the individual instance level, this means all solutions/non-solutions have the same shape (typically list or matrix).

Then, I propose we use the same learning bias as the rostering work, namely constraints of the form:

lb <= expr <= ub

where we can determine lb/ub by simply taking the min/max of the 'exprs'.

In terms of 'exprs' on which to find a comparison constraint, we consider two cases: unary variables, and pairs of variables. We can consider _all_ possible unary and pairs, that is going to be computationally easy.

Then, what is the grammar for unary expressions?

Unary(v) := v | abs(v) | power(v,2) | modulo(v,2)

why these? well, there are two unary operators: abs() and -(). However, lb <= -v <= ub <-> -ub <= v <= -lb; so we can skip -(). Then, there are binary operators v+Const, v-Const, v\*Const, v/Const; however, again we can move those to lb/ub, so we skip it. Then the final two binary operators on integer variables in CPMpy are modulo(v, Const) and power(v, Const). Anecdotical evidence, which we _could_ validate in CPMpy, says that module 2 and power 2 are pretty much the only reasonable options (e.g. check for even, and distance-like). So, the above is sufficient for most cases I expect.

Then, what is the grammar for binary expressions?

I already listed the binary operators above; and we can apply the unary operators on top of those too, so we get:

Binary(v,w) := (v != w) | Unary(Bin(v,w))     # so Bin(v,w) | abs(Bin(v,w)) | power(Bin(v,w),2) | modulo(Bin(v,w),2)
Bin(v,w) := Unary(v) + Unary(w) | Unary(v) - Unary(w) | Unary(w) - Unary(v) | Unlikely(v,w)
Unlikely(v,w) := Unary(v) \* Unary(w) | Unary(v) / Unary(w) | Unary(w) / Unary(v) | modulo(Unary(v), Unary(w)) | modulo(Unary(w), Unary(v)) | power(Unary(v), Unary(w)) | power(Unary(w), Unary(v))

I've grouped together the unlikely ones. Given that we are in the integer domain, expressions like v\*w or v\*abs(w) or v/power(w,2) are really really really unlikely. But Samuel wanted a general grammar : ) Also, v >= w is equivalent to v - w >= 0.

If we ignore the unlikely ones, we have 1+4\*( 3\*(4\*4) ) = 193 candidate expressions on which to compute lb/ubs. We could reduce this to 1+4\*3 if we drop the inner 'Unary' expressions, e.g. it will most likely be v - w and not v - abs(w) or v - modulo(w,2)... I would drop it, but Samuel wanted... ; )


What don't we have?

- we dont have expr != Const; but, really, if we do, most of them would be spurious I think...
- if the variables are Boolean, we don't have Boolean operators (the 7 types I saw were fully int)
- we dont have sum(allvars) or sum(rows) or groups like that...
    -> note that 'sum()', 'min()', 'max()' are the only n-ary mathematical operators in CPMpy... so we could add All(vars) = Unary(sum(vars)) | Unary(min(vars)) | Unary(max(vars))
- we don't have the Element constraint list[v] <-> Element(list, v), which I think is fine, it needs external data
- we don't support external data, e.g. weighted sum or magic constants. wsum(vars, weights) is easy to check in case a 'weight' vector of equal size is given as part of the 'input'...

