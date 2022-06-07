// dinosaur code always takes place in brightly-lit areas but that's just because i don't need to
// show off all the time
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
using ll = int64_t;
#define rep(i, a, n) for (int i = a; i < (int)(n); i++)
#define per(i, a, n) for (int i = (n)-1; i >= (int)(a); i--)
template <typename T, typename U>
ostream& operator<<(ostream& _s, pair<T, U> _t) {
    _s << "(" << _t.first << "," << _t.second << ")";
    return _s;
}
template <typename T, typename U>
istream& operator>>(istream& _s, pair<T, U>& _t) {
    _s >> _t.first >> _t.second;
    return _s;
}
template <typename T>
ostream& operator<<(ostream& _s, vector<T> _t) {
    rep(i, 0, _t.size()) _s << (i ? " " : "") << _t[i];
    return _s;
}
template <typename T>
istream& operator>>(istream& _s, vector<T>& _t) {
    rep(i, 0, _t.size()) _s >> _t[i];
    return _s;
}
template <typename T>
ostream& operator<<(ostream& _s, vector<vector<T>> _t) {
    rep(i, 0, _t.size()) {
        _s << i << ": " << _t[i] << endl;
    }
    return _s;
}

// https://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
vector<vector<bool>> comb(int N, int K) {
    vector<bool> bitmask(K, 1);  // K leading 1's
    bitmask.resize(N, 0);        // N-K trailing 0's

    vector<vector<bool>> res;

    // print integers and permute bitmask
    do {
        res.push_back(bitmask);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

    return res;
}

bool increase(vector<int>& indices, int max_val) {
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i] < max_val) {
            indices[i]++;
            return true;
        } else {
            indices[i] = 0;
        }
    }

    return false;
}

vector<int> evaluate(vector<int> a, int n, int m, int n_rows) {
    vector<int> results;
    vector<vector<int>> rows(n_rows, vector<int>(m));

    for (int mask = 0; mask < (1 << m); mask++) {
        for (int i = 0; i < m; i++) {
            int group = ((mask & (1 << (i % m))) > 0);

            // cout << mask << "  " << i << " " << group << " " << j << endl;

            for (int row = 0; row < n_rows; row++) {
                rows[row][i] = a[i + m * group + 2 * m * row];
            }
        }
        // cout << endl;

        // for (int row = 0; row < n_rows; row++) {
        //     for (int i = 0; i < m; i++) {
        //         cout << rows[row][i] << " ";
        //     }
        //     cout << endl;
        // }

        int res = 0;

        for (int row = 0; row < n_rows; row++) {
            sort(rows[row].begin(), rows[row].end());
            for (int i = n; i < m; i++) {
                res += rows[row][i];
            }
        }
        // cout << "->" << res << endl;
        results.push_back(res);
    }

    return results;
}

void solve(int n, int m, int n_rows) {
    const vector<int> values = {0, 1};

    int n_cells = 2 * m * n_rows;
    vector<int> indices(n_cells, 0);
    vector<int> a(n_cells, 0);

    // vector<pair<vector<int>, vector<int>>> groups = {{
    //     {0, 1, 2},
    // }};

    int iii = 0;
    int n_solutions = 0;

    do {
        iii++;
        // if (iii < 8) {
        //     continue;
        // }
        // if (iii > 8) {
        //     break;
        // }

        for (int i = 0; i < n_cells; i++) {
            a[i] = values[indices[i]];
        }
        vector<int> results = evaluate(a, n, m, n_rows);

        bool good = true;
        if (results[0] <= results[1])
            good = false;
        for (int i = 1; i < results.size(); i++) {
            if (results[i] != results[1])
                good = false;
        }

        if (good) {
            n_solutions++;
            cout << "found it:" << endl;
            for (int row = 0; row < n_rows; row++) {
                for (int i = 0; i < m * 2; i++) {
                    if (i == m)
                        cout << "  ";
                    cout << a[i + 2 * m * row] << " ";
                }
                cout << endl;
            }

            cout << "results: " << results << endl;

            // break;
        }
        // cout << rows << endl;
    } while (increase(indices, values.size() - 1));

    cout << "#solutions: " << n_solutions << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    solve(1, 3, 3);

    // auto x = evaluate({0,0,0,1,1,1,1,1,1,0,0,0}, 1, 3, 2);
    // cout << x << endl;
}

// for 1:3 sparsity:
// 0 0 0   0 1 1
// 0 0 0   1 0 0
// 1 1 1   0 0 0
// results: 2 3 3 3 3 3 3 3

// 1 1 0   0 1 1
// 1 0 1   0 1 0
// 0 1 0   1 0 0
// results: 5 4 4 4 4 4 4 4


1 1 0
1 0 1
0 1 1