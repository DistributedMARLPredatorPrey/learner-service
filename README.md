# Learner Service

The Learner Service is responsible for training two distinct MARL models: one for the Predators agents and another for the Preys agents.

The provided implementation uses the MADDPG algorithm for training and leverages a distributed Replay Buffer to batch agent experiences.
It also updates the agents' policies within a distributed Predator-Prey environment through a Publish/Subscribe channel.

## Usage

To deploy `learner-service` alongside the other microservices, follow the instructions provided in the [Bootstrap](https://github.com/DistributedMARLPredatorPrey/bootstrap) repository.

## License

Learner Service is licensed under the GNU v3.0 License. See the [LICENSE](./LICENSE) file for details.

## Author

- Luca Fabri ([w-disaster](https://github.com/w-disaster))
