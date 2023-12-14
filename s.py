values = [-10.00, 50, 90]

value_min = min(values) 
value_max = max(values) 
translation = (value_min + value_max) / 2 
dilation = value_max - value_min  
normalized_values = [0.5+((value - translation) / dilation) for value in values]
print(normalized_values)