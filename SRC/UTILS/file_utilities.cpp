#include "UTILS/file_utilities.h"

#include <fstream>

using namespace std;

bool file_exists(string filename)
{
  ifstream file(filename.c_str());
  return file.good();
}
