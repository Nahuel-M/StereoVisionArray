#pragma once
#include <iostream>
#include <fstream>
#include <vector>


std::vector<std::string> splitString(const std::string& str, const std::string& delimiter);

void saveReference(std::string inputPath);
