#ifndef BIG_FLOAT_H
#define BIG_FLOAT_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdint>

// 高精度浮点数类
template<int Precision = 50>  // 模板参数指定小数部分精度
class BigFloat {
private:
    bool is_negative;               // 符号位
    std::vector<int> integer_part;  // 整数部分（高位在前）
    std::vector<int> decimal_part;  // 小数部分（高位在前，长度固定为Precision）

    // 移除整数部分前导零
    void trim_leading_zeros() {
        while (integer_part.size() > 1 && integer_part[0] == 0) {
            integer_part.erase(integer_part.begin());
        }
        // 处理全零情况
        if (integer_part.empty()) {
            integer_part.push_back(0);
        }
    }

    // 移除小数部分尾随零（但保持Precision长度）
    void trim_trailing_zeros() {
        int last_non_zero = Precision - 1;
        while (last_non_zero >= 0 && decimal_part[last_non_zero] == 0) {
            last_non_zero--;
        }
        // 保留到最后一个非零位，其余补零
        for (int i = last_non_zero + 1; i < Precision; i++) {
            decimal_part[i] = 0;
        }
    }

    // 转换为内部表示的大整数（用于乘法）
    std::vector<int> to_big_integer(int& decimal_shift) const {
        decimal_shift = Precision;  // 小数部分长度
        std::vector<int> result = integer_part;
        result.insert(result.end(), decimal_part.begin(), decimal_part.end());
        // 移除前导零
        while (!result.empty() && result[0] == 0) {
            result.erase(result.begin());
        }
        if (result.empty()) result.push_back(0);
        return result;
    }

    // 从大整数和小数点位置构造
    void from_big_integer(const std::vector<int>& big_int, int decimal_shift) {
        int total_len = big_int.size();
        int int_len = total_len - decimal_shift;
        
        // 分离整数和小数部分
        if (int_len <= 0) {
            integer_part = {0};
            // 小数部分需要补前导零
            int leading_zeros = -int_len;
            decimal_part.clear();
            for (int i = 0; i < leading_zeros && i < Precision; i++) {
                decimal_part.push_back(0);
            }
            // 复制有效数字
            for (int i = 0; i < total_len && (leading_zeros + i) < Precision; i++) {
                decimal_part.push_back(big_int[i]);
            }
            // 补全剩余精度
            while (decimal_part.size() < Precision) {
                decimal_part.push_back(0);
            }
        } else {
            // 整数部分
            integer_part = std::vector<int>(big_int.begin(), big_int.begin() + int_len);
            // 小数部分
            decimal_part.clear();
            for (int i = int_len; i < total_len && i - int_len < Precision; i++) {
                decimal_part.push_back(big_int[i]);
            }
            // 补全剩余精度
            while (decimal_part.size() < Precision) {
                decimal_part.push_back(0);
            }
        }
        trim_leading_zeros();
    }

    // 大整数乘法
    static std::vector<int> multiply_big_integers(const std::vector<int>& a, const std::vector<int>& b) {
        int a_len = a.size();
        int b_len = b.size();
        std::vector<int> result(a_len + b_len, 0);

        // 乘法运算
        for (int i = a_len - 1; i >= 0; i--) {
            for (int j = b_len - 1; j >= 0; j--) {
                int product = a[i] * b[j];
                int sum = product + result[i + j + 1];
                result[i + j + 1] = sum % 10;
                result[i + j] += sum / 10;
            }
        }

        // 移除前导零
        while (!result.empty() && result[0] == 0) {
            result.erase(result.begin());
        }
        if (result.empty()) result.push_back(0);
        return result;
    }

    // 大整数比较（a > b 返回true）
    static bool greater_than(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return a[i] > b[i];
        }
        return false;  // 相等
    }

    // 大整数减法（a >= b）
    static std::vector<int> subtract_big_integers(const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result;
        int borrow = 0;
        int a_idx = a.size() - 1;
        int b_idx = b.size() - 1;

        while (a_idx >= 0 || b_idx >= 0) {
            int a_val = (a_idx >= 0) ? a[a_idx--] : 0;
            int b_val = (b_idx >= 0) ? b[b_idx--] : 0;

            a_val -= borrow;
            borrow = 0;

            if (a_val < b_val) {
                a_val += 10;
                borrow = 1;
            }

            result.push_back(a_val - b_val);
        }

        // 移除尾随零并反转
        while (result.size() > 1 && result.back() == 0) {
            result.pop_back();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    // 大整数除法（返回商和余数）
    static std::pair<std::vector<int>, std::vector<int>> divide_big_integers(
        const std::vector<int>& dividend, const std::vector<int>& divisor) {
        
        if (greater_than(divisor, dividend)) {
            return {{0}, dividend};  // 商为0，余数为被除数
        }

        std::vector<int> quotient;
        std::vector<int> remainder;

        for (int digit : dividend) {
            remainder.push_back(digit);
            // 移除前导零
            while (remainder.size() > 1 && remainder[0] == 0) {
                remainder.erase(remainder.begin());
            }

            // 计算当前位的商
            int count = 0;
            while (!greater_than(divisor, remainder)) {
                remainder = subtract_big_integers(remainder, divisor);
                count++;
            }
            quotient.push_back(count);
        }

        // 移除商的前导零
        while (quotient.size() > 1 && quotient[0] == 0) {
            quotient.erase(quotient.begin());
        }
        return {quotient, remainder};
    }

public:
    // 构造函数
    BigFloat() : is_negative(false) {
        integer_part = {0};
        decimal_part.resize(Precision, 0);
    }

    BigFloat(double value) : BigFloat(std::to_string(value)) {}

    BigFloat(const std::string& s) : is_negative(false) {
        std::string str = s;
        decimal_part.resize(Precision, 0);

        // 处理符号
        if (!str.empty() && str[0] == '-') {
            is_negative = true;
            str = str.substr(1);
        }

        // 分离整数和小数部分
        size_t dot_pos = str.find('.');
        std::string int_str, dec_str;

        if (dot_pos == std::string::npos) {
            int_str = str;
            dec_str = "";
        } else {
            int_str = str.substr(0, dot_pos);
            dec_str = str.substr(dot_pos + 1);
        }

        // 解析整数部分
        if (int_str.empty() || int_str == "0") {
            integer_part = {0};
        } else {
            for (char c : int_str) {
                if (isdigit(c)) {
                    integer_part.push_back(c - '0');
                }
            }
            if (integer_part.empty()) integer_part = {0};
        }

        // 解析小数部分
        for (int i = 0; i < Precision && i < dec_str.size(); i++) {
            if (isdigit(dec_str[i])) {
                decimal_part[i] = dec_str[i] - '0';
            } else {
                decimal_part[i] = 0;
            }
        }

        trim_leading_zeros();
    }

    // 转换为字符串
    std::string to_string() const {
        std::stringstream ss;
        if (is_negative) ss << "-";

        // 整数部分
        for (int digit : integer_part) {
            ss << digit;
        }

        // 小数部分
        ss << ".";
        for (int digit : decimal_part) {
            ss << digit;
        }

        return ss.str();
    }

    // 加法运算
    BigFloat operator+(const BigFloat& other) const {
        BigFloat result;
        if (is_negative == other.is_negative) {
            // 同号相加
            result.is_negative = is_negative;
            add_abs(other, result);
        } else {
            // 异号相减
            if (compare_abs(other) >= 0) {
                result.is_negative = is_negative;
                sub_abs(other, result);
            } else {
                result.is_negative = other.is_negative;
                other.sub_abs(*this, result);
            }
        }
        return result;
    }

    // 减法运算
    BigFloat operator-(const BigFloat& other) const {
        BigFloat neg_other = other;
        neg_other.is_negative = !neg_other.is_negative;
        return *this + neg_other;
    }

    // 乘法运算
    BigFloat operator*(const BigFloat& other) const {
        BigFloat result;
        result.is_negative = is_negative != other.is_negative;

        // 转换为大整数并记录小数点位置
        int shift1, shift2;
        std::vector<int> a = to_big_integer(shift1);
        std::vector<int> b = other.to_big_integer(shift2);

        // 大整数乘法
        std::vector<int> product = multiply_big_integers(a, b);

        // 计算总小数点偏移
        int total_shift = shift1 + shift2;
        result.from_big_integer(product, total_shift);

        return result;
    }

    // 除法运算
    BigFloat operator/(const BigFloat& other) const {
        if (other.is_zero()) {
            throw std::invalid_argument("Division by zero");
        }

        BigFloat result;
        result.is_negative = is_negative != other.is_negative;

        // 转换为大整数（扩展精度用于除法）
        int shift1, shift2;
        std::vector<int> dividend = to_big_integer(shift1);
        std::vector<int> divisor = other.to_big_integer(shift2);

        // 扩展被除数以提高精度（补充Precision位）
        dividend.insert(dividend.end(), Precision, 0);

        // 执行除法
        auto [quotient, remainder] = divide_big_integers(dividend, divisor);

        // 计算小数点位置
        int total_shift = shift1 - shift2 + Precision;
        result.from_big_integer(quotient, total_shift);

        return result;
    }

    // 判断是否为零
    bool is_zero() const {
        if (integer_part.size() != 1 || integer_part[0] != 0) {
            return false;
        }
        for (int d : decimal_part) {
            if (d != 0) return false;
        }
        return true;
    }

    // 赋值运算
    BigFloat& operator=(const BigFloat& other) {
        is_negative = other.is_negative;
        integer_part = other.integer_part;
        decimal_part = other.decimal_part;
        return *this;
    }

    // 与基础类型的运算
    BigFloat operator*(double scalar) const {
        return *this * BigFloat(scalar);
    }

    BigFloat operator+(double scalar) const {
        return *this + BigFloat(scalar);
    }

    BigFloat operator-(double scalar) const {
        return *this - BigFloat(scalar);
    }

    BigFloat operator/(double scalar) const {
        return *this / BigFloat(scalar);
    }

    // 比较绝对值大小
    int compare_abs(const BigFloat& other) const {
        // 比较整数部分长度
        if (integer_part.size() > other.integer_part.size()) return 1;
        if (integer_part.size() < other.integer_part.size()) return -1;

        // 比较整数部分每一位
        for (size_t i = 0; i < integer_part.size(); i++) {
            if (integer_part[i] > other.integer_part[i]) return 1;
            if (integer_part[i] < other.integer_part[i]) return -1;
        }

        // 比较小数部分
        for (int i = 0; i < Precision; i++) {
            if (decimal_part[i] > other.decimal_part[i]) return 1;
            if (decimal_part[i] < other.decimal_part[i]) return -1;
        }

        return 0;
    }
        static BigFloat<Precision> epsilon();    // 机器epsilon
    static BigFloat<Precision> max();        // 最大值
    static BigFloat<Precision> min();        // 最小值
    // 在BigFloat类中添加：
BigFloat<Precision> exp() const {
    // 简化实现：泰勒展开（仅示例，需优化精度）
    BigFloat<Precision> result(1.0);
    BigFloat<Precision> term(1.0);
    BigFloat<Precision> x = *this;
    for (int n = 1; n <= 50; ++n) {
        term = term * x / BigFloat<Precision>(n);
        result = result + term;
    }
    return result;
}

BigFloat<Precision> sqrt() const {
    // 简化实现：牛顿迭代法
    if (is_negative) throw std::domain_error("sqrt of negative number");
    BigFloat<Precision> x = *this;
    BigFloat<Precision> guess(x);  // 初始猜测值
    BigFloat<Precision> eps = epsilon();
    for (int i = 0; i < 50; ++i) {
        BigFloat<Precision> new_guess = (guess + x / guess) * BigFloat<Precision>("0.5");
        if ((new_guess - guess).abs() < eps) break;
        guess = new_guess;
    }
    return guess;
}

BigFloat<Precision> abs() const {
    BigFloat<Precision> res = *this;
    res.is_negative = false;
    return res;
}
private:
    // 绝对值加法
    void add_abs(const BigFloat& other, BigFloat& result) const {
        // 小数部分从后往前加
        int carry = 0;
        for (int i = Precision - 1; i >= 0; i--) {
            int sum = decimal_part[i] + other.decimal_part[i] + carry;
            result.decimal_part[i] = sum % 10;
            carry = sum / 10;
        }

        // 整数部分从后往前加
        size_t i = integer_part.size() - 1;
        size_t j = other.integer_part.size() - 1;
        result.integer_part.clear();

        while (i < integer_part.size() || j < other.integer_part.size() || carry > 0) {
            int sum = carry;
            if (i < integer_part.size()) sum += integer_part[i--];
            if (j < other.integer_part.size()) sum += other.integer_part[j--];
            
            result.integer_part.push_back(sum % 10);
            carry = sum / 10;
        }

        // 反转整数部分（恢复高位在前）
        std::reverse(result.integer_part.begin(), result.integer_part.end());
        result.trim_leading_zeros();
    }

    // 绝对值减法（假设this >= other）
    void sub_abs(const BigFloat& other, BigFloat& result) const {
        // 小数部分减法
        int borrow = 0;
        for (int i = Precision - 1; i >= 0; i--) {
            int diff = decimal_part[i] - other.decimal_part[i] - borrow;
            borrow = 0;
            if (diff < 0) {
                diff += 10;
                borrow = 1;
            }
            result.decimal_part[i] = diff;
        }

        // 整数部分减法
        size_t i = integer_part.size() - 1;
        size_t j = other.integer_part.size() - 1;
        result.integer_part.clear();

        while (i < integer_part.size() || j < other.integer_part.size()) {
            int diff = -borrow;
            borrow = 0;
            if (i < integer_part.size()) diff += integer_part[i--];
            if (j < other.integer_part.size()) diff -= other.integer_part[j--];
            
            if (diff < 0) {
                diff += 10;
                borrow = 1;
            }
            result.integer_part.push_back(diff);
        }

        // 反转整数部分（恢复高位在前）
        std::reverse(result.integer_part.begin(), result.integer_part.end());
        result.trim_leading_zeros();
    }

};

// 最大值：整数部分多位9 + 小数部分全9
// 模板参数为int，与类声明的int类型匹配
template<int Precision>
BigFloat<Precision> BigFloat<Precision>::epsilon() {  // 返回类型为BigFloat<Precision>
    if (Precision <= 0) {
        throw std::invalid_argument("Precision must be positive for epsilon()");
    }
    std::string epsilon_str = "0.";
    epsilon_str += std::string(Precision - 1, '0');
    epsilon_str += '1';
    return BigFloat<Precision>(epsilon_str);  // 构造函数需支持字符串参数
}

// 同理修复max()和min()的定义
template<int Precision>
BigFloat<Precision> BigFloat<Precision>::max() {
    if (Precision <= 0) {
        throw std::invalid_argument("Precision must be positive for max()");
    }
    std::string max_str = std::string(100, '9') + "." + std::string(Precision, '9');
    return BigFloat<Precision>(max_str);
}

template<int Precision>
BigFloat<Precision> BigFloat<Precision>::min() {
    return epsilon();  // 调用同类的epsilon()
}
// 输入输出流支持
template<int P>
std::ostream& operator<<(std::ostream& os, const BigFloat<P>& bf) {
    os << bf.to_string();
    return os;
}

// 为BigFloat<P>特化Eigen的NumTraits
template <int P>
struct Eigen::NumTraits<BigFloat<P>> : Eigen::NumTraits<double> {
    // 基础类型定义
    typedef BigFloat<P> Real;               // 实数类型（自身）
    typedef BigFloat<P> NonInteger;         // 非整数类型（自身）
    typedef BigFloat<P> Nested;             // 嵌套类型

    // 数值特性标志
    enum {
        IsInteger = 0,                      // BigFloat是浮点数，非整数
        IsSigned = 1,                       // 支持负数
        RequireInitialization = 1,          // 需要显式初始化
        ReadCost = 10,                      // 读取成本（相对值，高精度类型较高）
        AddCost = 20,                       // 加法成本
        MulCost = 100                       // 乘法成本（高精度运算成本更高）
    };

    // 精度相关函数（需与BigFloat的实现匹配）
    static Real epsilon() {
        return BigFloat<P>::epsilon();  // 返回BigFloat的机器epsilon
    }

    static Real dummy_precision() {
        // 用于比较时的最小容差（根据精度P调整）
        return BigFloat<P>("1e-16");  // 示例值，需适配实际精度
    }

    // 数值范围函数
    static Real highest() {
        return BigFloat<P>::max();  // 最大值
    }

    static Real lowest() {
        return BigFloat<P>::min();  // 最小值（最负）
    }

    // 十进制有效数字位数
    static int digits10() {
        // 假设P表示二进制精度位数，近似转换为十进制位数（1二进制位≈0.3十进制位）
        return static_cast<int>(P * 0.3);
    }
};


#endif // BIG_FLOAT_H