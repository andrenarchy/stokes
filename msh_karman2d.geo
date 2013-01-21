Point(1) = {0.0, 0.0, 0.0}; // lower left
Point(2) = {2.2, 0.0, 0.0}; // lower right
Point(3) = {2.2, 0.41, 0.0}; // upper right
Point(4) = {0.0, 0.41, 0.0}; // upper left
Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left
Line Loop(1) = {1, 2, 3, 4}; // box

Point(5) = {0.2, 0.2, 0.0}; // circle center
Point(6) = {0.15, 0.2, 0.0}; // circle left
Point(7) = {0.2, 0.15, 0.0}; // circle bottom
Point(8) = {0.25, 0.2, 0.0}; // circle right
Point(9) = {0.2, 0.25, 0.0}; // circle top
Circle(5) = {6, 5, 7}; // lower left circle arc
Circle(6) = {7, 5, 8}; // lower right circle arc
Circle(7) = {8, 5, 9}; // upper right circle arc
Circle(8) = {9, 5, 6}; // upper left circle arc
Line Loop(2) = {5, 6, 7, 8}; // circle

Plane Surface(1) = { 1, 2 }; // box without circle
