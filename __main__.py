from kaggle_environments import make
from constants import ROOT_PATH, SETTINGS
from agent.agent import halite_agent

if __name__ == "__main__":
    print("Making environment")
    env = make("halite", debug=True)

    print("Running agent")
    env.run([halite_agent, "random", "random", "random"])

    print("Saving replay")
    render_options = SETTINGS["replay"]
    game_path = ROOT_PATH / "games"
    game_path.mkdir(parents=False, exist_ok=True)
    game_output_path = game_path / "output.html"
    f = open(game_output_path, "w")
    f.write(env.render(**render_options))
