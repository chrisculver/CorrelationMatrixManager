CXX = g++
CXXFLAGS = -std=c++17 -Wall -I./INCLUDE/

UTILITIES = string_utilities.cpp file_utilities.cpp diagrams.cpp

SOURCE = $(UTILITIES) 
OBJECTS = $(SOURCE:.cpp=.o) 

all: util

util: $(OBJECTS) 
	$(CXX) $(CXXFLAGS) -o convert_diagramnames_toinput $(OBJECTS) $(LINKS) convert_diagramnames_toinput.cpp $(LINKS)
	$(CXX) $(CXXFLAGS) -o compare_files $(OBJECTS) $(LINKS) compare_files.cpp $(LINKS)

clean:
	rm $(OBJECTS) convert_diagramnames_toinput compare_files
