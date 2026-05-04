function [Neighbors] = Neighbor(L, x, y)
above = mod(x - 2, L) + 1;
below = mod(x, L) + 1;
left = mod(y - 2, L) + 1;
right = mod(y, L) + 1;
Neighbors = [above, right, left, below];
end
