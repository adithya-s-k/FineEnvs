"""Train the standalone blackbox-opencode environment with the reference async recipe."""
from data_agent_env import opencode_agent_turns
from standalone_comparison import ScheduledOpenCodeFactory
from train_harbor_multi import main

if __name__ == "__main__":
    main(session_factory_class=ScheduledOpenCodeFactory, agent_turn_selector=opencode_agent_turns)
