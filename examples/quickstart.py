
from reexpastar.wa_star_cr import WeightedAStarCR, WeightedAStarAR, WeightedAStarNR, Grid

if __name__ == "__main__":
    g = Grid(30, 30, walls={(15, y) for y in range(30)} - {(15, 10)})
    start, goal = (0,0), (29,29)
    h = g.manhattan(goal); is_goal = lambda s: s == goal

    for name, eng in [
        ("NR", WeightedAStarNR(start, is_goal, g.neighbors, h, weight=1.5)),
        ("AR", WeightedAStarAR(start, is_goal, g.neighbors, h, weight=1.5)),
        ("CR(r=0.3)", WeightedAStarCR(start, is_goal, g.neighbors, h, weight=1.5, r=0.3)),
    ]:
        path, cost, st = eng.run_to_first_goal()
        print(f"{name}: cost={cost}, expansions={st.expansions}, reopens={st.reopens}")
