from kaggle_environments import make

if __name__ == "__main__":
    env = make("halite", debug=True)
    env.run(["./agent/agent.py", "random", "random", "random"])
    f = open("./games/output.html", "w")
    f.write(env.render(mode="html", width=800, height=600))