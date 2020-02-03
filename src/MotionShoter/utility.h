#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <string>
#include <vector>

namespace ms
{

enum InputType {
    VIDEO,
    DATASET,
    IMAGES
};

void ReadImageGTFiles(const std::string& folder, std::vector<std::string>& gtFiles);
void ReadImageFiles(const std::string& folder, std::vector<std::string>& files);


}  // namespace ms

#endif  // UTILITY_HPP
