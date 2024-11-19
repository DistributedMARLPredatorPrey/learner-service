.. Learner Service documentation master file, created by
   sphinx-quickstart on Sat Jan 13 16:59:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Learner Service's documentation!
===========================================

**Learner Service** is responsible for training two distinct MARL models: one for the Predators agents and another for the Preys agents.

The provided implementation uses the MADDPG algorithm for training and leverages a distributed Replay Buffer to batch agent experiences.
It also updates the agents' policies within a distributed Predator-Prey environment through a Publish/Subscribe channel.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
