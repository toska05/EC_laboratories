#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

struct Move {
    int type;
    int a;
    int b;
};

int deltaInter(const Instance& I, const vector<int>& s, int i, int u) {
    int m = s.size();
    int v = s[i];
    int a = s[(i + m - 1) % m];
    int b = s[(i + 1) % m];
    return I.d[a][u] + I.d[u][b] - I.d[a][v] - I.d[v][b] + I.cost[u] - I.cost[v];
}

int deltaNodes(const Instance& I, const vector<int>& s, int i, int j) {
    int m = s.size();
    if (i == 0 && j == m - 1) {
        int x = s[m - 1], y = s[0];
        int p = s[m - 2], q = s[1];
        return I.d[p][y] + I.d[x][q] - I.d[p][x] - I.d[y][q];
    }
    int a = s[i], b = s[j];
    int pi = s[(i + m - 1) % m], ni = s[(i + 1) % m];
    int pj = s[(j + m - 1) % m], nj = s[(j + 1) % m];
    if (j == i + 1) {
        return I.d[pi][b] + I.d[a][nj] - I.d[pi][a] - I.d[b][nj];
    }
    return I.d[pi][b] + I.d[b][ni] + I.d[pj][a] + I.d[a][nj]
         - I.d[pi][a] - I.d[a][ni] - I.d[pj][b] - I.d[b][nj];
}

int deltaEdges(const Instance& I, const vector<int>& s, int i, int j) {
    int m = s.size();
    int a = s[i], b = s[i + 1];
    int c = s[j], e = s[(j + 1) % m];
    return I.d[a][c] + I.d[b][e] - I.d[a][b] - I.d[c][e];
}

int evalMove(const Instance& I, const vector<int>& s, const vector<int>& outside, const Move& mv, bool useEdges) {
    if (mv.type == 0) return deltaInter(I, s, mv.a, outside[mv.b]);
    return useEdges ? deltaEdges(I, s, mv.a, mv.b) : deltaNodes(I, s, mv.a, mv.b);
}

void applyMove(vector<int>& s, vector<int>& outside, const Move& mv, bool useEdges) {
    if (mv.type == 0) {
        swap(s[mv.a], outside[mv.b]);
    } else if (useEdges) {
        reverse(s.begin() + mv.a + 1, s.begin() + mv.b + 1);
    } else {
        swap(s[mv.a], s[mv.b]);
    }
}

vector<Move> buildMoves(int m, int outsideSize, bool useEdges) {
    vector<Move> moves;
    for (int i = 0; i < m; i++) {
        for (int t = 0; t < outsideSize; t++) {
            moves.push_back({0, i, t});
        }
    }
    for (int i = 0; i < m - 1; i++) {
        for (int j = i + 1; j < m; j++) {
            if (useEdges && (j - i < 2 || (i == 0 && j == m - 1))) continue;
            moves.push_back({1, i, j});
        }
    }
    return moves;
}

vector<int> outsideNodes(const Instance& I, const vector<int>& s) {
    vector<char> in(I.n, 0);
    for (int v : s) in[v] = 1;
    vector<int> outside;
    for (int j = 0; j < I.n; j++) {
        if (!in[j]) outside.push_back(j);
    }
    return outside;
}

long long lastEvals = 0;
int lastIters = 0;

void steepestSearch(const Instance& I, vector<int>& s, bool useEdges) {
    vector<int> outside = outsideNodes(I, s);
    vector<Move> moves = buildMoves(s.size(), outside.size(), useEdges);
    lastEvals = 0;
    lastIters = 0;
    while (true) {
        int bestDelta = 0;
        int bestIdx = -1;
        for (int idx = 0; idx < (int)moves.size(); idx++) {
            int delta = evalMove(I, s, outside, moves[idx], useEdges);
            if (delta < bestDelta) {
                bestDelta = delta;
                bestIdx = idx;
            }
        }
        lastEvals += moves.size();
        if (bestIdx < 0) break;
        applyMove(s, outside, moves[bestIdx], useEdges);
        lastIters++;
    }
}

void greedySearch(const Instance& I, vector<int>& s, bool useEdges, mt19937& rng) {
    vector<int> outside = outsideNodes(I, s);
    vector<Move> moves = buildMoves(s.size(), outside.size(), useEdges);
    int total = moves.size();
    bool improved = true;
    lastEvals = 0;
    lastIters = 0;
    while (improved) {
        improved = false;
        for (int idx = 0; idx < total; idx++) {
            uniform_int_distribution<int> dist(idx, total - 1);
            int r = dist(rng);
            swap(moves[idx], moves[r]);
            int delta = evalMove(I, s, outside, moves[idx], useEdges);
            lastEvals++;
            if (delta < 0) {
                applyMove(s, outside, moves[idx], useEdges);
                lastIters++;
                improved = true;
                break;
            }
        }
    }
}

struct Candidates {
    vector<vector<int>> intra;
    vector<vector<int>> inter;
};

vector<vector<int>> nearestLists(const Instance& I, int K, bool withCost) {
    vector<vector<int>> res(I.n);
    for (int a = 0; a < I.n; a++) {
        vector<int> others;
        for (int b = 0; b < I.n; b++) {
            if (b != a) others.push_back(b);
        }
        auto key = [&](int b) { return I.d[a][b] + (withCost ? I.cost[b] : 0); };
        stable_sort(others.begin(), others.end(), [&](int x, int y) { return key(x) < key(y); });
        others.resize(min(K, (int)others.size()));
        res[a] = others;
    }
    return res;
}

Candidates buildCandidates(const Instance& I, int kIntra, int kInter, bool costInIntra, bool costInInter) {
    Candidates c;
    c.intra = nearestLists(I, kIntra, costInIntra);
    c.inter = nearestLists(I, kInter, costInInter);
    return c;
}

void candidateSearch(const Instance& I, vector<int>& s, const Candidates& C) {
    int m = s.size();
    vector<int> pos(I.n, -1);
    for (int i = 0; i < m; i++) pos[s[i]] = i;
    lastEvals = 0;
    lastIters = 0;

    while (true) {
        int bestDelta = 0;
        int bestType = -1;
        int bestX = -1, bestY = -1;

        auto tryEdges = [&](int x, int y) {
            int lo = min(x, y), hi = max(x, y);
            if (hi - lo < 2 || (lo == 0 && hi == m - 1)) return;
            int delta = deltaEdges(I, s, lo, hi);
            lastEvals++;
            if (delta < bestDelta) {
                bestDelta = delta;
                bestType = 1;
                bestX = lo;
                bestY = hi;
            }
        };
        auto tryInter = [&](int x, int u) {
            int delta = deltaInter(I, s, x, u);
            lastEvals++;
            if (delta < bestDelta) {
                bestDelta = delta;
                bestType = 0;
                bestX = x;
                bestY = u;
            }
        };

        for (int i = 0; i < m; i++) {
            int a = s[i];
            int prev = (i + m - 1) % m;
            int next = (i + 1) % m;
            for (int b : C.intra[a]) {
                int j = pos[b];
                if (j < 0) continue;
                tryEdges(i, j);
                tryEdges(prev, (j + m - 1) % m);
            }
            for (int b : C.inter[a]) {
                if (pos[b] >= 0) continue;
                tryInter(next, b);
                tryInter(prev, b);
            }
        }

        if (bestType < 0) break;
        lastIters++;
        if (bestType == 0) {
            pos[s[bestX]] = -1;
            s[bestX] = bestY;
            pos[bestY] = bestX;
        } else {
            reverse(s.begin() + bestX + 1, s.begin() + bestY + 1);
            for (int t = bestX + 1; t <= bestY; t++) pos[s[t]] = t;
        }
    }
}

struct LMove {
    int delta;
    int type;
    int a, b, c, e;
    long long id;
};

struct LMWorse {
    bool operator()(const LMove& x, const LMove& y) const {
        if (x.delta != y.delta) return x.delta > y.delta;
        return x.id > y.id;
    }
};

long long lmMisses = 0;

void lmSearch(const Instance& I, vector<int>& s) {
    int m = s.size();
    int n = I.n;
    vector<int> pos(n, -1);
    for (int i = 0; i < m; i++) pos[s[i]] = i;
    vector<LMove> LM;
    vector<LMove> held;
    auto push = [&](const LMove& mv) {
        LM.push_back(mv);
        push_heap(LM.begin(), LM.end(), LMWorse());
    };
    long long nextId = 0;
    lastEvals = 0;
    lastIters = 0;

    auto nxt = [&](int v) { return s[(pos[v] + 1) % m]; };
    auto prv = [&](int v) { return s[(pos[v] + m - 1) % m]; };
    auto adjacent = [&](int u, int v) {
        return pos[u] >= 0 && pos[v] >= 0 && (nxt(u) == v || nxt(v) == u);
    };

    auto evalInter = [&](int x, int u) {
        int p = prv(x), q = nxt(x);
        int delta = I.d[p][u] + I.d[u][q] - I.d[p][x] - I.d[x][q] + I.cost[u] - I.cost[x];
        lastEvals++;
        if (delta < 0) push({delta, 0, x, u, p, q, nextId++});
    };
    auto evalEdges = [&](int a, int b, int c, int e) {
        int delta = I.d[a][c] + I.d[b][e] - I.d[a][b] - I.d[c][e];
        lastEvals++;
        if (delta < 0) push({delta, 1, a, b, c, e, nextId++});
    };

    auto evaluateAll = [&]() {
        LM.clear();
        held.clear();
        vector<int> outside;
        for (int v = 0; v < n; v++) {
            if (pos[v] < 0) outside.push_back(v);
        }
        for (int i = 0; i < m; i++) {
            for (int u : outside) evalInter(s[i], u);
        }
        for (int i = 0; i < m - 1; i++) {
            for (int j = i + 1; j < m; j++) {
                if (j - i < 2 || (i == 0 && j == m - 1)) continue;
                int a = s[i], b = s[i + 1], c = s[j], e = s[(j + 1) % m];
                evalEdges(a, b, c, e);
                evalEdges(a, b, e, c);
            }
        }
    };

    auto evaluateNew = [&](const vector<int>& T, const vector<pair<int, int>>& newEdges, int freed) {
        for (int t : T) {
            for (int u = 0; u < n; u++) {
                if (pos[u] < 0) evalInter(t, u);
            }
        }
        if (freed >= 0) {
            for (int i = 0; i < m; i++) {
                int y = s[i];
                if (find(T.begin(), T.end(), y) == T.end()) evalInter(y, freed);
            }
        }
        for (int k = 0; k < (int)newEdges.size(); k++) {
            int u = newEdges[k].first, v = newEdges[k].second;
            for (int i = 0; i < m; i++) {
                int c = s[i], e = s[(i + 1) % m];
                if (c == u || c == v || e == u || e == v) continue;
                bool seen = false;
                for (int kk = 0; kk < k; kk++) {
                    int f = newEdges[kk].first, g = newEdges[kk].second;
                    if ((c == f && e == g) || (c == g && e == f)) seen = true;
                }
                if (seen) continue;
                evalEdges(u, v, c, e);
                evalEdges(u, v, e, c);
            }
        }
    };

    auto reversePart = [&](int i, int j) {
        int lo, hi;
        if (i < j) {
            lo = i + 1;
            hi = j;
        } else {
            lo = j + 1;
            hi = i;
        }
        reverse(s.begin() + lo, s.begin() + hi + 1);
        for (int t = lo; t <= hi; t++) pos[s[t]] = t;
    };

    evaluateAll();
    bool afterRefresh = false;
    while (true) {
        bool applied = false;
        while (!LM.empty()) {
            pop_heap(LM.begin(), LM.end(), LMWorse());
            LMove mv = LM.back();
            LM.pop_back();
            if (mv.type == 0) {
                int x = mv.a, u = mv.b;
                if (pos[x] < 0 || pos[u] >= 0) continue;
                int p = prv(x), q = nxt(x);
                if (!((p == mv.c && q == mv.e) || (p == mv.e && q == mv.c))) continue;
                int i = pos[x];
                s[i] = u;
                pos[x] = -1;
                pos[u] = i;
                lastIters++;
                evaluateNew({p, u, q}, {{p, u}, {u, q}}, x);
                applied = true;
                break;
            } else {
                int a = mv.a, b = mv.b, c = mv.c, e = mv.e;
                if (!adjacent(a, b) || !adjacent(c, e)) continue;
                bool dir1 = (nxt(a) == b);
                bool dir2 = (nxt(c) == e);
                if (dir1 != dir2) {
                    held.push_back(mv);
                    continue;
                }
                if (dir1) reversePart(pos[a], pos[c]);
                else reversePart(pos[b], pos[e]);
                lastIters++;
                evaluateNew({a, b, c, e}, {{a, c}, {b, e}}, -1);
                applied = true;
                break;
            }
        }
        for (const LMove& h : held) push(h);
        held.clear();
        if (applied) {
            if (afterRefresh) lmMisses++;
            afterRefresh = false;
            continue;
        }
        if (afterRefresh) break;
        evaluateAll();
        afterRefresh = true;
    }
}

struct Method {
    string name;
    function<vector<int>(const Instance&, int, mt19937&)> func;
    bool perStartNode;
};

struct LSConfig {
    string name;
    int kind;
    bool useEdges;
    bool greedyStart;
    int kIntra, kInter;
    bool costInIntra;
    bool costInInter;
    bool inMainTable;
};

struct Stats {
    long long minV, maxV;
    double avgV;
    double minT, maxT, avgT;
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
        {"nn_any_regret", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 0.0, false); }, true},
        {"greedy_cycle_regret", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 0.0, true); }, true},
        {"nn_any_weighted", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 1.0, false); }, true},
        {"greedy_cycle_weighted", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 1.0, true); }, true},
        {"nn_any_weighted_1_2", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 2.0, false); }, true},
        {"greedy_cycle_weighted_1_2", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 1.0, 2.0, true); }, true},
        {"nn_any_weighted_2_1", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 2.0, 1.0, false); }, true},
        {"greedy_cycle_weighted_2_1", [](const Instance& I, int s, mt19937&) { return greedyRegret(I, s, 2.0, 1.0, true); }, true}
    };

    vector<LSConfig> configs = {
        {"steepest_nodes_random", 0, false, false, 0, 0, false, false, true},
        {"steepest_nodes_greedy", 0, false, true, 0, 0, false, false, true},
        {"steepest_edges_random", 0, true, false, 0, 0, false, false, true},
        {"steepest_edges_greedy", 0, true, true, 0, 0, false, false, true},
        {"greedy_nodes_random", 1, false, false, 0, 0, false, false, true},
        {"greedy_nodes_greedy", 1, false, true, 0, 0, false, false, true},
        {"greedy_edges_random", 1, true, false, 0, 0, false, false, true},
        {"greedy_edges_greedy", 1, true, true, 0, 0, false, false, true},
        {"candidate_edges_random", 2, true, false, 10, 10, false, true, true},
        {"candidate_edges_greedy", 2, true, true, 10, 10, false, true, true},
        {"lm_edges_random", 3, true, false, 0, 0, false, false, true},
        {"lm_edges_greedy", 3, true, true, 0, 0, false, false, true}
    };

    const int RANDOM_RUNS = 200;
    mt19937 rng(12345);
    mt19937 rngLS(777);

    filesystem::create_directories("results");
    ofstream summary("results/results.csv");
    summary << "instance,method,min,max,avg,time_min_ms,time_max_ms,time_avg_ms\n";
    ofstream lsStats("results/ls_stats.csv");
    lsStats << "instance,method,k_intra,k_inter,cost_in_intra,cost_in_inter,avg_iterations,avg_evaluated_moves,min,max,avg,time_min_ms,time_max_ms,time_avg_ms\n";

    vector<string> order;
    map<string, map<string, Stats>> table;
    vector<string> instNames;

    using Clock = chrono::steady_clock;

    for (const string& file : files) {
        Instance inst = loadInstance(file);
        string iname = instanceName(file);
        instNames.push_back(iname);
        cout << "instance " << iname << " n=" << inst.n << " k=" << inst.k << endl;

        auto record = [&](const string& name, const Stats& st, const vector<int>& bestSol, bool inMain) {
            cout << setw(28) << name << "  obj: " << fixed << setprecision(1) << st.avgV
                 << " (" << st.minV << " - " << st.maxV << ")  time[ms]: "
                 << setprecision(3) << st.avgT << " (" << st.minT << " - " << st.maxT << ")" << endl;
            if (!inMain) return;
            summary << iname << "," << name << "," << st.minV << "," << st.maxV << ","
                    << fixed << setprecision(1) << st.avgV << ","
                    << setprecision(4) << st.minT << "," << st.maxT << "," << st.avgT << "\n";
            ofstream out("results/best_" + iname + "_" + name + ".txt");
            for (int v : bestSol) out << v << "\n";
            table[name][iname] = st;
            if (find(order.begin(), order.end(), name) == order.end()) order.push_back(name);
        };

        int bestGreedyIdx = -1;
        double bestGreedyAvg = 1e18;

        for (int mi = 0; mi < (int)methods.size(); mi++) {
            const Method& m = methods[mi];
            int runs = m.perStartNode ? inst.n : RANDOM_RUNS;
            Stats st{LLONG_MAX, LLONG_MIN, 0, 1e18, 0, 0};
            vector<int> bestSol;
            double sum = 0, sumT = 0;
            for (int r = 0; r < runs; r++) {
                int start = m.perStartNode ? r : 0;
                auto t0 = Clock::now();
                vector<int> sol = m.func(inst, start, rng);
                auto t1 = Clock::now();
                double ms = chrono::duration<double, milli>(t1 - t0).count();
                long long val = objective(inst, sol);
                sum += val;
                sumT += ms;
                st.minT = min(st.minT, ms);
                st.maxT = max(st.maxT, ms);
                st.maxV = max(st.maxV, val);
                if (val < st.minV) {
                    st.minV = val;
                    bestSol = sol;
                }
            }
            st.avgV = sum / runs;
            st.avgT = sumT / runs;
            record(m.name, st, bestSol, true);
            if (m.perStartNode && st.avgV < bestGreedyAvg) {
                bestGreedyAvg = st.avgV;
                bestGreedyIdx = mi;
            }
        }

        cout << "best greedy construction for starts: " << methods[bestGreedyIdx].name << endl;

        mt19937 rngStart(2024);
        vector<vector<int>> randomStarts;
        for (int r = 0; r < RANDOM_RUNS; r++) randomStarts.push_back(randomSolution(inst, 0, rngStart));

        for (const LSConfig& c : configs) {
            Candidates cand;
            double prepMs = 0;
            if (c.kind == 2) {
                auto p0 = Clock::now();
                cand = buildCandidates(inst, c.kIntra, c.kInter, c.costInIntra, c.costInInter);
                prepMs = chrono::duration<double, milli>(Clock::now() - p0).count();
            }
            Stats st{LLONG_MAX, LLONG_MIN, 0, 1e18, 0, 0};
            vector<int> bestSol;
            double sum = 0, sumT = 0, sumTotalT = 0, sumEvals = 0, sumIters = 0;
            for (int r = 0; r < RANDOM_RUNS; r++) {
                auto tStart = Clock::now();
                vector<int> sol = c.greedyStart ? methods[bestGreedyIdx].func(inst, r, rngLS)
                                                : randomStarts[r];
                auto t0 = Clock::now();
                if (c.kind == 0) steepestSearch(inst, sol, c.useEdges);
                else if (c.kind == 1) greedySearch(inst, sol, c.useEdges, rngLS);
                else if (c.kind == 2) candidateSearch(inst, sol, cand);
                else {
                    lmSearch(inst, sol);
                }
                auto t1 = Clock::now();
                double ms = chrono::duration<double, milli>(t1 - t0).count();
                sumTotalT += chrono::duration<double, milli>(t1 - tStart).count();
                sumEvals += lastEvals;
                sumIters += lastIters;
                long long val = objective(inst, sol);
                sum += val;
                sumT += ms;
                st.minT = min(st.minT, ms);
                st.maxT = max(st.maxT, ms);
                st.maxV = max(st.maxV, val);
                if (val < st.minV) {
                    st.minV = val;
                    bestSol = sol;
                }
            }
            st.avgV = sum / RANDOM_RUNS;
            st.avgT = sumT / RANDOM_RUNS;
            record(c.name, st, bestSol, c.inMainTable);
            cout << "   " << setw(26) << c.name << fixed << setprecision(1)
                 << "  iterations: " << sumIters / RANDOM_RUNS
                 << "  evaluated moves: " << sumEvals / RANDOM_RUNS
                 << setprecision(3) << "  LS time: " << st.avgT << " ms"
                 << "  with construction: " << sumTotalT / RANDOM_RUNS << " ms";
            if (c.kind == 2) cout << "  candidate lists: " << prepMs << " ms";
            if (c.kind == 3) cout << "  LM misses: " << lmMisses;
            cout << endl;
            lsStats << iname << "," << c.name << "," << c.kIntra << "," << c.kInter << ","
                    << c.costInIntra << "," << c.costInInter << ","
                    << fixed << setprecision(1) << sumIters / RANDOM_RUNS << "," << sumEvals / RANDOM_RUNS << ","
                    << st.minV << "," << st.maxV << "," << st.avgV << ","
                    << setprecision(4) << st.minT << "," << st.maxT << "," << st.avgT << "\n";
        }
    }

    ofstream tobj("results/table_objective.csv");
    ofstream ttime("results/table_time_ms.csv");
    tobj << "method";
    ttime << "method";
    for (const string& in : instNames) {
        tobj << "," << in;
        ttime << "," << in;
    }
    tobj << "\n";
    ttime << "\n";
    for (const string& name : order) {
        tobj << name;
        ttime << name;
        for (const string& in : instNames) {
            const Stats& st = table[name][in];
            tobj << fixed << setprecision(1) << ",\"" << st.avgV << " (" << st.minV << " - " << st.maxV << ")\"";
            ttime << fixed << setprecision(2) << ",\"" << st.avgT << " (" << st.minT << " - " << st.maxT << ")\"";
        }
        tobj << "\n";
        ttime << "\n";
    }
    return 0;
}