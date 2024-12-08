CXX = g++

CXXFLAGS = -Wall -Wextra -std=c++11

SRCS = array_utilities.cpp main.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = main

$(TARGET) : $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS)