from agent.ship import HaliteShip

class TestHaliteShip:
    def __init__(self):



    def test_get_dir_to(self):
        test_ship = HaliteShip([10, 10])
        assert test_ship.get_dir_to([10, 11])