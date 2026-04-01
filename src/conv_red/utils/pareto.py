
def find_pareto(d):
    front = []

    for i in range(len(d)):
        dom = True
        for j in range(len(d)):
            if d[j][0] > d[i][0] and d[j][1] > d[i][1]:
                dom = False
                break
        if dom: front.append(i)

    return front

