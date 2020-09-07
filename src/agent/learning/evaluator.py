def evaluate_board(board: "HaliteBoard"):
    """
    Evaluates a given board state
    :param board: HaliteBoard to evaluate for the current player
    """

    opponent_halite = sum([x.halite for x in board.opponents.values()])
    avg_opponent_halite = opponent_halite / 3
    return board.player.halite - avg_opponent_halite


from src.agent.board.board import HaliteBoard
