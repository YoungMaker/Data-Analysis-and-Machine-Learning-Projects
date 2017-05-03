from itertools import combinations
import random
from os import path as pth

def create_point_list(num_points):
    all_waypoints = []

    for x in xrange(num_points):
        all_waypoints.append("p" + str(x))

    return all_waypoints

def create_tsv_file(all_waypoints, waypoints_file):

    waypoint_distances = {}
    waypoint_durations = {}

    for (waypoint1, waypoint2) in combinations(all_waypoints, 2):

        #assign random distances

        #print("%s | %s\n" % (waypoint1, waypoint2))

        waypoint_distances[frozenset([waypoint1, waypoint2])] = random.randint(8, 1.6093e+7)
        waypoint_durations[frozenset([waypoint1, waypoint2])] = 0

    #print waypoint_distances

    print("Saving Waypoints")
    with open(waypoints_file, "w") as out_file:
        out_file.write("\t".join(["waypoint1",
                                  "waypoint2",
                                  "distance_m",
                                  "duration_s"]))

        for (waypoint1, waypoint2) in waypoint_distances.keys():
            out_file.write("\n" +
                           "\t".join([waypoint1,
                                      waypoint2,
                                      str(waypoint_distances[frozenset([waypoint1, waypoint2])]),
                                      str(waypoint_durations[frozenset([waypoint1, waypoint2])])]))


if __name__ == '__main__':
    random.seed()
    i = 400
    fname = "my-waypoints_auto"

    while i < 1000:
        fname = "my-waypoints_auto" + str(i) + ".tsv"
        if not pth.isfile(fname):
            create_tsv_file(create_point_list(i), "my-waypoints_auto" + str(i) + ".tsv")
        i+= 200
