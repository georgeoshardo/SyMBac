import pymunk
import time
space = pymunk.Space()
space.gravity = 0, -100

body = pymunk.Body(1,1666)
body.position = 50, 100

space.add(body)
while True:
    space.step(0.02)
    print(body.position)
    time.sleep(0.5)