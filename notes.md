1. What is gradient

If i nudge w up a tiny bit, how much deos the loss change?

+gradient = nudging w up makes loss worse, so move w down
-gradient = nudging w up makes loss better, so move w up
large gradient = loss is very sensitive to w, so take bigger step
gradient ≈ 0, we found bottom, stop 

new w = w -learning_rate * gradient 

2. What we do with gradients in this project

We have adjustable numbers: w(seconds per token) and b (starting cost). Traning means finding the w and b that make the loss smallest 

We need something that tells us for each param:
    Which direction should i move it, and how much deos it matter?


3. Observation and research

Fast model generated x= 201 tokens and it took y=0.8821 s

Starting guess: Here i pick w = 0.002, b= 0.1 

predictor = w * tokens + b
= 0.002 * 201 + 0.1 
=0.502 s

error = prediction - y 
=0.502 - 0.8821
= -0.3801 (guessed too fast)

loss = 1/2 (error)^2
=1/2 * (-0.3801)^2 
=0.072238

now lets nudge w by a tiny amount, so w= 0.00201

predictor = 0.00201 *201 +0.1 
=0.50401 s 

error = 0.50401 - 0.8821
= -0.37809

loss = 1/2(-0.37809)^2
= 0.071476

loss decreased while w go up it means -gradient and w moves up

gradient ≈ change in loss / change in w = −0.000762 / 0.00001 ≈ −76.2


However, this nudge and measure method is the numerical gradient. It works but it needs an extra calculation for every params, which is too slow for real networks. So we want a formula

4. Formula: Chain rule

The loss depends on w through a chain: 

w -> prediction changes -> loss changes 
When you change w it doesnt touch the loss directly.
To know how w affects the loss, find out
1. How much does the predeciton move when w moves?
2. How much does the loss move when the prediction move? 
3. Multiply them 

w -> pred = w *x + b -> loss = 1/2(pred-y)^2

Rate 1: how much prediction moves when w moves
pred = w * 201 + b
lets raise w by 0.001

before: 0.002 * 201 = 0.402
afrer: 0.003 * 201 = 0.603

prediction moved by 0.201

so for every 1 unit of w, prediction moves 201. 

Rate 2: how much the loss moves when prediction moves 
loss = 1/2(error)^2 

rn error is -0.38

before: error is -0.3801, loss = 0.072238
afrer : error is -0.3791, loss = 0.071859 

loss moved by -0.00038, which is -0.38 * 0.001

so for every 1 unit of prediction, losss moves -0.38 which is error itself 

Multiply the two rates: 
w  rate 1: 201    rate 2: -0.38  total = rate 1 * rate 2   201 * -0.38 = -76.4
b  rate 1: 1      rate 2: -0.38  total = rate 1 * rate 2   1 * -0.38 = -0.38

So, 
gradient of w = error * x  
gradient of b = error 
