#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <functional>

using namespace std;

struct Instance {
    int n;
    int k;
    vector<int> cost;
    vector<vector<int>> d;
};

Instance loadInstance(const string& path) {
    ifstream in(path);
    if (!in) {
        cerr << "unable to open the file: " << path << endl;
        exit(1);
    }
    vector<double> xs, ys;
    vector<int> cs;
    string line;
    while (getline(in, line)) {
        for (char& ch : line) {
            if (ch == ';' || ch == ',') ch = ' ';
        }
        istringstream ss(line);
        double x, y, c;
        if (ss >> x >> y >> c) {
            xs.push_back(x);
            ys.push_back(y);
            cs.push_back((int)c);
        }
    }
    Instance inst;
    inst.n = xs.size();
    inst.k = (inst.n + 1) / 2;
    inst.cost = cs;
    inst.d.assign(inst.n, vector<int>(inst.n, 0));
    for (int i = 0; i < inst.n; i++) {
        for (int j = 0; j < inst.n; j++) {
            double dx = xs[i] - xs[j];
            double dy = ys[i] - ys[j];
            inst.d[i][j] = (int)llround(sqrt(dx * dx + dy * dy));
        }
    }
    return inst;
}

long long objective(const Instance& I, const vector<int>& s) {
    long long total = 0;
    int m = s.size();
    for (int i = 0; i < m; i++) {
        total += I.d[s[i]][s[(i + 1) % m]] + I.cost[s[i]];
    }
    return total;
}

vector<int> randomSolution(const Instance& I, int start, mt19937& rng) {
    vector<int> all(I.n);
    for (int i = 0; i < I.n; i++) all[i] = i;
    shuffle(all.begin(), all.end(), rng);
    all.resize(I.k);
    return all;
}

vector<int> nearestEnd(const Instance& I, int start, mt19937& rng) {
    vector<int> path;
    path.push_back(start);
    vector<char> used(I.n, 0);
    used[start] = 1;
    while ((int)path.size() < I.k) {
        int last = path.back();
        int bestJ = -1;
        int bestVal = INT_MAX;
        for (int j = 0; j < I.n; j++) {
            if (used[j]) continue;
            int v = I.d[last][j] + I.cost[j];
            if (v < bestVal) {
                bestVal = v;
                bestJ = j;
            }
        }
        path.push_back(bestJ);
        used[bestJ] = 1;
    }
    return path;
}

vector<int> nearestAnyPosition(const Instance& I, int start, mt19937& rng) {
    vector<int> path;
    path.push_back(start);
    vector<char> used(I.n, 0);
    used[start] = 1;
    while ((int)path.size() < I.k) {
        int size = path.size();
        int bestJ = -1;
        int bestPos = -1;
        int bestVal = INT_MAX;
        for (int j = 0; j < I.n; j++) {
            if (used[j]) continue;
            for (int p = 0; p <= size; p++) {
                int delta;
                if (p == 0) {
                    delta = I.d[j][path[0]];
                } else if (p == size) {
                    delta = I.d[path[size - 1]][j];
                } else {
                    int a = path[p - 1];
                    int b = path[p];
                    delta = I.d[a][j] + I.d[j][b] - I.d[a][b];
                }
                delta += I.cost[j];
                if (delta < bestVal) {
                    bestVal = delta;
                    bestJ = j;
                    bestPos = p;
                }
            }
        }
        path.insert(path.begin() + bestPos, bestJ);
        used[bestJ] = 1;
    }
    return path;
}

vector<int> greedyCycle(const Instance& I, int start, mt19937& rng) {
    vector<int> cycle;
    cycle.push_back(start);
    vector<char> used(I.n, 0);
    used[start] = 1;
    int second = -1;
    int bestVal = INT_MAX;
    for (int j = 0; j < I.n; j++) {
        if (used[j]) continue;
        int v = I.d[start][j] + I.cost[j];
        if (v < bestVal) {
            bestVal = v;
            second = j;
        }
    }
    cycle.push_back(second);
    used[second] = 1;
    while ((int)cycle.size() < I.k) {
        int m = cycle.size();
        int bestJ = -1;
        int bestPos = -1;
        int best = INT_MAX;
        for (int j = 0; j < I.n; j++) {
            if (used[j]) continue;
            for (int i = 0; i < m; i++) {
                int a = cycle[i];
                int b = cycle[(i + 1) % m];
                int delta = I.d[a][j] + I.d[j][b] - I.d[a][b] + I.cost[j];
                if (delta < best) {
                    best = delta;
                    bestJ = j;
                    bestPos = i + 1;
                }
            }
        }
        cycle.insert(cycle.begin() + bestPos, bestJ);
        used[bestJ] = 1;
    }
    return cycle;
}

// Insertion cost of node j at position p.
// cycleMode: p in [0, m-1], node goes between seq[p] and seq[(p+1)%m] (closed cycle).
// path mode: p in [0, m], node goes before seq[p]; p=0 and p=m extend the ends of the open path.
int insertionDelta(const Instance& I, const vector<int>& seq, int j, int p, bool cycleMode) {
    int m = seq.size();
    int delta;
    if (cycleMode) {
        int a = seq[p];
        int b = seq[(p + 1) % m];
        delta = I.d[a][j] + I.d[j][b] - I.d[a][b];
    } else if (p == 0) {
        delta = I.d[j][seq[0]];
    } else if (p == m) {
        delta = I.d[seq[m - 1]][j];
    } else {
        int a = seq[p - 1];
        int b = seq[p];
        delta = I.d[a][j] + I.d[j][b] - I.d[a][b];
    }
    return delta + I.cost[j];
}

// Greedy construction with a weighted criterion: score = wRegret * regret - wCost * bestDelta.
// regret = (2nd best insertion cost) - (best insertion cost) of the node.
// wRegret=1, wCost=0 gives pure 2-regret.
// cycleMode=true  -> base is greedy cycle (start with 2 nodes, insert into cycle edges)
// cycleMode=false -> base is nearest neighbor any position (start with 1 node, insert at any path position)
vector<int> greedyRegret(const Instance& I, int start, double wRegret, double wCost, bool cycleMode) {
    vector<int> seq;
    seq.push_back(start);
    vector<char> used(I.n, 0);
    used[start] = 1;

    if (cycleMode) {
        int second = -1;
        int bestVal = INT_MAX;
        for (int j = 0; j < I.n; j++) {
            if (used[j]) continue;
            int v = I.d[start][j] + I.cost[j];
            if (v < bestVal) {
                bestVal = v;
                second = j;
            }
        }
        seq.push_back(second);
        used[second] = 1;
    }

    while ((int)seq.size() < I.k) {
        int m = seq.size();
        int positions = cycleMode ? m : m + 1;
        int chosenJ = -1;
        int chosenPos = -1;
        double bestScore = -1e18;
        for (int j = 0; j < I.n; j++) {
            if (used[j]) continue;
            int best = INT_MAX, second = INT_MAX, bestPos = -1;
            for (int p = 0; p < positions; p++) {
                int delta = insertionDelta(I, seq, j, p, cycleMode);
                if (delta < best) {
                    second = best;
                    best = delta;
                    bestPos = p;
                } else if (delta < second) {
                    second = delta;
                }
            }
            int regret = second - best;
            double score = wRegret * regret - wCost * best;
            if (score > bestScore) {
                bestScore = score;
                chosenJ = j;
                chosenPos = bestPos;
            }
        }
        int insertAt = cycleMode ? chosenPos + 1 : chosenPos;
        seq.insert(seq.begin() + insertAt, chosenJ);
        used[chosenJ] = 1;
    }
    return seq;
}

struct Method {
    string name;
    function<vector<int>(const Instance&, int, mt19937&)> func;
    bool perStartNode;
};

string instanceName(const string& path) {
    size_t slash = path.find_last_of("/\\");
    string name = (slash == string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != string::npos) name = name.substr(0, dot);
    return name;
}

int main(int argc, char* argv[]) {
    vector<string> files;
    for (int i = 1; i < argc; i++) files.push_back(argv[i]);
    if (files.empty()) {
        files.push_back("../data/TSPA.csv");
        files.push_back("../data/TSPB.csv");
    }

    vector<Method> methods = {
        {"random", randomSolution, false},
        {"nn_end", nearestEnd, true},
        {"nn_any", nearestAnyPosition, true},
        {"greedy_cycle", greedyCycle, true},
        // 2-regret only
        {"nn_any_regret", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 0.0, false); }, true},
        {"greedy_cycle_regret", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 0.0, true); }, true},
        // weighted sum: regret and best objective change, equal weights
        {"nn_any_weighted", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 1.0, false); }, true},
        {"greedy_cycle_weighted", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 1.0, true); }, true},
        // weight experiments (regret weight : cost weight)
        {"nn_any_weighted_1_2", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 2.0, false); }, true},
        {"greedy_cycle_weighted_1_2", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 2.0, true); }, true},
        {"nn_any_weighted_2_1", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 2.0, 1.0, false); }, true},
        {"greedy_cycle_weighted_2_1", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 2.0, 1.0, true); }, true}
    };

    const int RANDOM_RUNS = 200;
    mt19937 rng(12345);

    filesystem::create_directories("results");
    ofstream summary("results/results.csv");
    summary << "instance,method,min,max,avg\n";

    for (const string& file : files) {
        Instance inst = loadInstance(file);
        string iname = instanceName(file);
        cout << "instance " << iname << " n=" << inst.n << " k=" << inst.k << endl;

        for (const Method& m : methods) {
            int runs = m.perStartNode ? inst.n : RANDOM_RUNS;
            long long best = LLONG_MAX;
            long long worst = LLONG_MIN;
            double sum = 0;
            vector<int> bestSol;

            for (int r = 0; r < runs; r++) {
                int start = m.perStartNode ? r : 0;
                vector<int> sol = m.func(inst, start, rng);
                long long val = objective(inst, sol);
                sum += val;
                if (val > worst) worst = val;
                if (val < best) {
                    best = val;
                    bestSol = sol;
                }
            }

            double avg = sum / runs;
            cout << setw(14) << m.name << "  min=" << best << "  max=" << worst
                 << "  avg=" << fixed << setprecision(1) << avg << endl;
            summary << iname << "," << m.name << "," << best << "," << worst << ","
                    << fixed << setprecision(1) << avg << "\n";

            ofstream out("results/best_" + iname + "_" + m.name + ".txt");
            for (size_t i = 0; i < bestSol.size(); i++) {
                out << bestSol[i] << "\n";
            }
        }
    }
    return 0;
}
