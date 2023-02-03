In this implementation, we separate the model into 3 graphs that connected by model.py
These graphs are:
1. representation graph (connection 1)
2. action graph (connection 1 but with action - ordered pairs as node?)
3. solution graph (connection 2)

However, the form of actions worth discussion. Currently we consider action as independent type.

By classifying physical action and mental action, we can consider physical action as a series of observation/representation and mental action as a function call. This will bring up some complexity.