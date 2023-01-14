import csv

with open("intraday/data/SPY.csv") as fr, open("intraday/data/NEW_SPY.csv","w", newline="") as fw:
    cr = csv.reader(fr,delimiter=",")
    cw = csv.writer(fw,delimiter=",")
    cw.writerow(next(cr))  # write title as-is
    cw.writerows(reversed(list(cr)))