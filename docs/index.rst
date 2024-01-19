.. Learner Service documentation master file, created by
   sphinx-quickstart on Sat Jan 13 16:59:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Learner Service's documentation!
===========================================

The Learner Service is responsible to train a Multi-Agent Reinforcement Learning (MARL) algorithm for a single
Predator-Prey application.

It samples batches of data from a centralized distributed Replay Buffer and updates the models consequently.

In particular it makes use of the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm,
composed of a centralized critic and decentralized actor networks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
