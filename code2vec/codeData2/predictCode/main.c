#include <iostream>
#include <vector>
#include <string>
using namespace std;
class N2Queens
{
public:
  int res = 0;
  int solution()
  {
    int N;
    cin >> N;
    vector<string> board(N, string(N, '.'));
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        cin >> board[i][j];
      }
    }
    push(0, board);
    return res;
  }
  void push(int row, vector<string> &board)
  {
    if (row == board.size())
    {
      res++;
      return;
    }
    int n = board[row].size();
    for (size_t col = 0; col < n ; col++)
    {
      if (board[row][col] == '0')
        continue;
      if (!ifVailed(board, row, col))
        continue;
      board[row][col] = 'Q';
      push(row + 1, board);
      board[row][col] = '1';
    }
  }

  bool ifVailed(vector<string> &board, int row, int col)
  {
    int n = board.size();

    for (size_t i = 0; i < n; i++)
    {
      if (board[i][col] == 'Q')
        return false;
    }
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
    {
      if (board[i][j] == 'Q')
        return false;
    }

    for (int i = row - 1, j = col + 1;
         i >= 0 && j < n; i--, j++)
    {
      if (board[i][j] == 'Q')
        return false;
    }
    return true;
  }
};

int main()
{
  N2Queens getRes;

  int a = getRes.solution();
  cout<<a;
}