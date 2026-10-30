Sometimes an AI chatbot answers in a blink.
Sometimes you stare at a spinner for 15 seconds.

I wanted to know why. So I tried to predict it.

The setup: I pretended to be the first ML engineer at a small startup that runs its own AI server on one Mac. It serves two models:
⚡ "fast": small, quick, okay answers
🧠 "quality": bigger, slower, better answers

The product team wants two things:
1. Show an ETA before you hit send ("≈ 2 seconds")
2. If you only want to wait 1.5 s and "quality" will take longer, send you to "fast"

I recorded 2,000 real requests: real human questions, random traffic, quiet hours and busy hours.

My one rule: Python + NumPy only. No scikit-learn, no PyTorch. I write the math myself.

Why make it harder? Because I could call .fit() for years and still not know what a gradient is.

So I started with pen and paper. One request: 201 tokens, 0.88 seconds. Make a guess, measure how wrong it is, nudge the guess a tiny bit, see if the error shrinks. Do that over and over and you have learning. It turns out the whole update is one line: gradient = error × input.

What the data showed me:

📏 Speed is a straight line. The fast model writes ~227 tokens per second, the quality one ~128.

🎲 The catch: nobody knows how long the answer will be before it's written. You only know the limit. Short limits get used up ~89% of the time. Long ones, under 5%.

🥶 Cold starts hurt. If a model sat idle for 20 s, it gets unloaded, and the next request is ~40% slower.

🚦 Crowds hurt more. Alone on the server: ~0.5 s. With 6+ other requests running: ~3.5 s.

And my first training run blew up, because I fed it raw token counts in the hundreds. Dividing by 1,000 fixed it. Small detail, big lesson.

Next: build the full predictor and beat "just guess the average" by at least 35%.

What would you want an AI app to tell you before you hit send?

#MachineLearning #LLM #Python #LearningInPublic
