#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cstring>

using namespace std;

class MatrixWrongSizeError {
};

class MatrixIndexError {
};

class MatrixIsDegenerateError {
};

template<typename T>
class Matrix {
private:
    int rowsCnt;
    int colsCnt;
    T **array;
protected:
    Matrix();

    virtual void assign(const Matrix<T> &);

    void new_memory(const int, const int);

    void free_memory();

    void add_row(const int, const int, const T);

    void row_swap(const int, const int);

public:
    Matrix(const int, const int);

    Matrix(const Matrix<T> &);

    virtual ~Matrix();

    Matrix<T> &operator=(const Matrix<T> &);

    Matrix<T> &operator+=(const Matrix<T> &);

    Matrix<T> &operator-=(const Matrix<T> &);

    Matrix<T> &operator*=(const Matrix<T> &);

    Matrix<T> &operator*=(const T);

    Matrix<T> &operator/=(const T);

    Matrix<T> &transpose();

    Matrix<T> getTransposed() const;

    int getRowsNumber() const;

    int getColumnsNumber() const;

    void set(const int, const int, const T);

    T operator()(const int, const int) const;

    T &operator()(const int, const int);

    Matrix<T> operator+(const Matrix<T> &) const;

    Matrix<T> operator-(const Matrix<T> &) const;

    Matrix<T> operator*(const Matrix<T> &) const;

    Matrix<T> operator*(const T &) const;

    template<typename C>
    friend Matrix<C> operator*(const C, const Matrix<C> &);

    Matrix<T> operator/(const T) const;

    template<typename C>
    friend std::istream &operator>>(std::istream &, Matrix<C> &);

    template<typename C>
    friend std::ostream &operator<<(std::ostream &, const Matrix<C> &);
};


template<typename T>
Matrix<T>::Matrix() {
    this->new_memory(1, 1);
}

template<typename T>
void Matrix<T>::assign(const Matrix<T> &other) {
    this->new_memory(other.rowsCnt, other.colsCnt);
    this->rowsCnt = other.rowsCnt;
    this->colsCnt = other.colsCnt;
    for (int row = 0; row < other.rowsCnt; row++) {
        for (int col = 0; col < other.colsCnt; col++) {
            this->array[row][col] = other.array[row][col];
        }
    }
}

template<typename T>
void Matrix<T>::free_memory() {
    for (int row = 0; row < this->rowsCnt; row++) {
        delete[] this->array[row];
    }
    delete[] this->array;
    this->rowsCnt = 0;
    this->colsCnt = 0;
}

template<typename T>
void Matrix<T>::new_memory(const int rowsCnt, const int colsCnt) {
    this->rowsCnt = rowsCnt;
    this->colsCnt = colsCnt;
    this->array = new T *[rowsCnt];
    for (int row = 0; row < rowsCnt; row++) {
        this->array[row] = new T[colsCnt];
        for (int col = 0; col < colsCnt; col++) {
            array[row][col] = 0;
        }
    }
}

template<typename T>
void Matrix<T>::set(const int row, const int col, const T value) {
    this->array[row][col] = value;
}

template<typename T>
void Matrix<T>::add_row(const int this_row, const int other_row, const T a) {
    for (int i = 0; i < this->colsCnt; i++) {
        this->set(this_row, i, this->array[this_row][i] + this->array[other_row][i] * a);
    }
}

template<typename T>
void Matrix<T>::row_swap(const int first_row, const int second_row) {
    for (int i = 0; i < this->colsCnt; i++) {
        T temp = this->array[first_row][i];
        this->array[first_row][i] = this->array[second_row][i];
        this->array[second_row][i] = -temp;
    }
}

template<typename T>
Matrix<T>::Matrix(const int rowsCnt, const int colsCnt) {
    this->new_memory(rowsCnt, colsCnt);
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &other) {
    this->assign(other);
}

template<typename T>
Matrix<T>::~Matrix() {
    for (int row = 0; row < this->rowsCnt; row++) {
        delete[] this->array[row];
    }
    delete[] this->array;
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
    if (this != &other) {
        this->free_memory();
        this->assign(other);
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &other) {
    *this = *this + other;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &other) {
    *this = *this - other;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &other) {
    *this = *this * other;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator*=(const T other) {
    *this = *this * other;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator/=(const T other) {
    *this = *this / other;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::transpose() {
    *this = this->getTransposed();
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::getTransposed() const {
    Matrix<T> result(this->colsCnt, this->rowsCnt);
    for (int row = 0; row < this->rowsCnt; row++) {
        for (int col = 0; col < this->colsCnt; col++) {
            result.array[col][row] = this->array[row][col];
        }
    }
    return result;
}

template<typename T>
int Matrix<T>::getRowsNumber() const {
    return this->rowsCnt;
}

template<typename T>
int Matrix<T>::getColumnsNumber() const {
    return this->colsCnt;
}

template<typename T>
T Matrix<T>::operator()(const int row, const int column) const {
    if (row >= this->rowsCnt || column >= this->colsCnt) {
        throw MatrixIndexError();
    }
    return this->array[row][column];
}

template<typename T>
T &Matrix<T>::operator()(const int row, const int column) {
    if (row >= this->rowsCnt || column >= this->colsCnt) {
        throw MatrixIndexError();
    }
    return this->array[row][column];
}

template<typename C>
std::istream &operator>>(std::istream &is, Matrix<C> &other) {
    for (int row = 0; row < other.rowsCnt; row++) {
        for (int col = 0; col < other.colsCnt; col++) {
            is >> other.array[row][col];
        }
    }
    return is;
}

template<typename C>
std::ostream &operator<<(std::ostream &os, const Matrix<C> &other) {
    for (int row = 0; row < other.rowsCnt; row++) {
        for (int col = 0; col < other.colsCnt; col++) {
            os << other.array[row][col] << ' ';
        }
        os << std::endl;
    }
    return os;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &a) const {
    if (this->rowsCnt != a.rowsCnt || this->colsCnt != a.colsCnt) {
        throw MatrixWrongSizeError();
    }
    Matrix<T> result(a.rowsCnt, a.colsCnt);
    for (int row = 0; row < a.rowsCnt; ++row) {
        for (int col = 0; col < a.colsCnt; ++col) {
            result.array[row][col] = a.array[row][col] + this->array[row][col];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &a) const {
    if (this->rowsCnt != a.rowsCnt || this->colsCnt != a.colsCnt) {
        throw MatrixWrongSizeError();
    }
    Matrix<T> result(a.rowsCnt, a.colsCnt);
    for (int row = 0; row < a.rowsCnt; ++row) {
        for (int col = 0; col < a.colsCnt; ++col) {
            result.array[row][col] = this->array[row][col] - a.array[row][col];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &b) const {
    if (this->colsCnt != b.rowsCnt) {
        throw MatrixWrongSizeError();
    }
    Matrix<T> result(this->rowsCnt, b.colsCnt);
    for (int row = 0; row < this->rowsCnt; ++row) {
        for (int col = 0; col < b.colsCnt; ++col) {
            result.array[row][col] = 0;
            for (int i = 0; i < this->colsCnt; i++) {
                result.array[row][col] += this->array[row][i] * b.array[i][col];
            }
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T &b) const {
    Matrix<T> result(this->rowsCnt, this->colsCnt);
    for (int row = 0; row < this->rowsCnt; ++row) {
        for (int col = 0; col < this->colsCnt; ++col) {
            result.array[row][col] = this->array[row][col] * b;
        }
    }
    return result;
}

template<typename C>
Matrix<C> operator*(const C a, const Matrix<C> &b) {
    return b * a;
}


template<typename T>
Matrix<T> Matrix<T>::operator/(const T b) const {
    Matrix<T> result(this->rowsCnt, this->colsCnt);
    for (int row = 0; row < this->rowsCnt; ++row) {
        for (int col = 0; col < this->colsCnt; ++col) {
            result.array[row][col] = this->array[row][col] / b;
        }
    }
    return result;
}

template<typename T>
class SquareMatrix : public Matrix<T> {
protected:
    void assign(const Matrix<T> &);

    SquareMatrix<T> my_minor(const int, const int) const;

public:
    SquareMatrix(const int);

    SquareMatrix(const Matrix<T> &);

    SquareMatrix<T> operator=(const SquareMatrix<T> &);

    SquareMatrix<T> &operator+=(const SquareMatrix<T> &);

    SquareMatrix<T> &operator-=(const SquareMatrix<T> &);

    SquareMatrix<T> &operator*=(const SquareMatrix<T> &);

    SquareMatrix<T> &operator*=(const T);

    SquareMatrix<T> &operator/=(const T);

    SquareMatrix<T> &transpose();

    SquareMatrix<T> getTransposed() const;

    SquareMatrix<T> operator+(const SquareMatrix<T> &) const;

    SquareMatrix<T> operator-(const SquareMatrix<T> &) const;

    SquareMatrix<T> operator*(const SquareMatrix<T> &) const;

    SquareMatrix<T> operator*(const T &) const;

    template<typename C>
    friend SquareMatrix<C> operator*(const C &, const SquareMatrix<C>);

    SquareMatrix<T> operator/(const T &) const;

    SquareMatrix<T> getInverse() const;

    SquareMatrix<T> &invert();

    T getDeterminant() const;

    int getSize() const;

    T getTrace() const;
};

template<typename T>
void SquareMatrix<T>::assign(const Matrix<T> &other) {
    this->Matrix<T>::assign(other);
}

template<typename T>
SquareMatrix<T>::SquareMatrix(const int size) {
    this->new_memory(size, size);
}

template<typename T>
SquareMatrix<T>::SquareMatrix(const Matrix<T> &other) {
    this->assign(other);
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::operator=(const SquareMatrix<T> &other) {
    if (this != &other) {
        this->free_memory();
        this->assign(other);
    }
    return *this;
}

template<typename T>
SquareMatrix<T> &SquareMatrix<T>::operator+=(const SquareMatrix<T> &other) {
    *this = *this + other;
    return *this;
}

template<typename T>
SquareMatrix<T> &SquareMatrix<T>::operator-=(const SquareMatrix<T> &other) {
    *this = *this - other;
    return *this;
}

template<typename T>
SquareMatrix<T> &SquareMatrix<T>::operator*=(const SquareMatrix<T> &other) {
    *this = *this * other;
    return *this;
}

template<typename T>
SquareMatrix<T> &SquareMatrix<T>::operator*=(const T a) {
    *this = *this * a;
    return *this;
}

template<typename T>
SquareMatrix<T> &SquareMatrix<T>::operator/=(const T a) {
    *this = *this / a;
    return *this;
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::getTransposed() const {
    return Matrix<T>::getTransposed();
}

template<typename T>
SquareMatrix<T> &SquareMatrix<T>::transpose() {
    *this = this->getTransposed();
    return *this;
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::operator+(const SquareMatrix<T> &other) const {
    return Matrix<T>::operator+(other);
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::operator-(const SquareMatrix<T> &other) const {
    return Matrix<T>::operator-(other);
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::operator*(const SquareMatrix<T> &other) const {
    return Matrix<T>::operator*(other);
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::operator*(const T &a) const {
    return Matrix<T>::operator*(a);
}

template<typename C>
SquareMatrix<C> operator*(const C &a, const SquareMatrix<C> matrix) {
    return matrix * a;
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::operator/(const T &a) const {
    return Matrix<T>::operator/(a);
}

template<typename T>
T SquareMatrix<T>::getDeterminant() const {
    SquareMatrix<T> temp = *this;
    bool flag = true;
    T det = T(1);
    for (int col = 0; col < temp.getColumnsNumber() && flag; col++) {
        int j = col;
        while (j < temp.getRowsNumber() && temp.operator()(j, col) == T(0)) {
            j++;
        }
        if (j == temp.getRowsNumber()) {
            flag = false;
        } else {
            if (j != col) {
                temp.row_swap(col, j);
            }
            for (int k = col + 1; k < temp.getRowsNumber(); k++) {
                T a = temp.operator()(k, col) / temp.operator()(col, col);
                temp.add_row(k, col, -a);
            }
        }
        det *= temp.operator()(col, col);
    }
    if (!flag) {
        det = T(0);
    }
    return det;
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::my_minor(const int without_row, const int without_col) const {
    SquareMatrix<T> result(this->getSize() - 1);
    for (int row = 0; row < this->getRowsNumber(); row++) {
        for (int col = 0; col < this->getColumnsNumber(); col++) {
            if (row != without_row && col != without_col) {
                int i = row;
                int j = col;
                if (row > without_row) {
                    i = row - 1;
                }
                if (col > without_col) {
                    j = col - 1;
                }
                result.operator()(i, j) = this->operator()(row, col);
            }
        }
    }
    return result;
}

template<typename T>
SquareMatrix<T> SquareMatrix<T>::getInverse() const {
    if (this->getDeterminant() == T(0)) {
        throw MatrixIsDegenerateError();
    }
    SquareMatrix<T> result(this->getSize());
    T this_det = this->getDeterminant();
    for (int row = 0; row < this->getRowsNumber(); row++) {
        for (int col = 0; col < this->getColumnsNumber(); col++) {
            result.operator()(row, col) = this->my_minor(row, col).getDeterminant() / this_det;
            if ((row + col) % 2 == 1) {
                result.operator()(row, col) *= -1;
            }
        }
    }
    result.transpose();
    return result;
}


template<typename T>
SquareMatrix<T> &SquareMatrix<T>::invert() {
    *this = this->getInverse();
    return *this;
}

template<typename T>
int SquareMatrix<T>::getSize() const {
    return this->getRowsNumber();
}

template<typename T>
T SquareMatrix<T>::getTrace() const {
    T sum = T(0);
    for (int row = 0; row < this->getRowsNumber(); row++) {
        sum += this->operator()(row, row);
    }
    return sum;
}

int gcd(int a, int b) {
    int t;
    while (b != 0) {
        t = b;
        b = a % b;
        a = t;
    }
    return abs(a);
}

class RationalDivisionByZero {
};

class Rational {
private:
    int p;
    int q; // is maintained to be positive

    void reduce() {
        int gcd1 = gcd(this->p, this->q);
        this->p /= gcd1;
        this->q /= gcd1;
        if (this->q < 0) {
            this->p *= -1;
            this->q *= -1;
        }
    }

public:
    Rational() : p(0), q(1) {}

    Rational(int p, int q) : p(p), q(q) {
        this->reduce();
    }

    Rational(int a) : p(a), q(1) {}

    Rational(const Rational &other) : p(other.p), q(other.q) {}

    int getNumerator() const {
        return this->p;
    }

    int getDenominator() const {
        return this->q;
    }

    Rational operator+(const Rational &rat) const {
        int gcd1 = gcd(this->q, rat.q);
        int denom = this->q / gcd1 * rat.q;
        int nom = rat.q / gcd1 * this->p + this->q / gcd1 * rat.p;
        Rational res(nom, denom);
        res.reduce();
        return res;
    }

    Rational operator+(const int a) const {
        return *this + Rational(a);
    }

    Rational operator-() const {
        return Rational(-this->p, this->q);
    }

    Rational operator-(const Rational &rat) const {
        return *this + (-rat);
    }

    Rational operator-(const int a) const {
        return *this - Rational(a);
    }

    Rational operator*(const Rational &rat) const {
        int gcd1 = gcd(this->p, rat.q), gcd2 = gcd(this->q, rat.p);
        Rational res(this->p / gcd1 * rat.p / gcd2, this->q / gcd2 * rat.q / gcd1);
        return res;
    }

    Rational operator*(const int a) const {
        return *this * Rational(a);
    }

    Rational operator/(const Rational &rat) const {
        if (rat.p == 0)
            throw RationalDivisionByZero();
        else {
            int gcd1 = gcd(this->p, rat.p), gcd2 = gcd(this->q, rat.q);
            Rational res(this->p / gcd1 * rat.q / gcd2, this->q / gcd2 * rat.p / gcd1);
            return res;
        }
    }

    Rational operator/(const int a) const {
        return *this / Rational(a);
    }

    Rational &operator+=(const Rational &rat) {
        int gcd1 = gcd(this->q, rat.q);
        this->p = rat.q / gcd1 * this->p + this->q / gcd1 * rat.p;
        this->q = this->q / gcd1 * rat.q;
        this->reduce();
        return *this;
    }

    Rational &operator-=(const Rational &rat) {
        int gcd1 = gcd(this->q, rat.q);
        this->p = rat.q / gcd1 * this->p - this->q / gcd1 * rat.p;
        this->q = this->q / gcd1 * rat.q;
        this->reduce();
        return *this;
    }

    Rational &operator*=(const Rational &rat) {
        int gcd1 = gcd(this->p, rat.q), gcd2 = gcd(this->q, rat.p);
        this->p = this->p / gcd1 * rat.p / gcd2;
        this->q = this->q / gcd2 * rat.q / gcd1;
        return *this;
    }

    Rational &operator*=(const int &a) {
        *this = *this * Rational(a);
        return *this;
    }

    Rational &operator/=(const Rational &rat) {
        int gcd1 = gcd(this->p, rat.p), gcd2 = gcd(this->q, rat.q);
        this->p = this->p / gcd1 * rat.q / gcd2;
        this->q = this->q / gcd2 * rat.p / gcd1;
        return *this;
    }

    Rational &operator/=(const int &a) {
        *this = *this / Rational(a);
        return *this;
    }

    bool operator==(const Rational &rat) const {
        return this->p == rat.p && this->q == rat.q;
    }

    bool operator!=(const Rational &rat) const {
        return !(*this == rat);
    }

    bool operator>(const Rational &rat) const {
        return (*this - rat).p > 0;
    }

    bool operator<(const Rational &rat) const {
        return (*this - rat).p < 0;
    }

    bool operator>=(const Rational &rat) const {
        return (*this - rat).p >= 0;
    }

    bool operator<=(const Rational &rat) const {
        return (*this - rat).p <= 0;
    }

    Rational &operator++() {
        this->p += this->q;
        return *this;
    }

    Rational operator++(int not_used) {
        Rational oldValue(*this);
        this->p += this->q;
        return oldValue;
    }

    Rational &operator--() {
        this->p -= this->q;
        return *this;
    }

    Rational operator--(int not_used) {
        Rational oldValue(*this);
        this->p -= this->q;
        return oldValue;
    }

    Rational operator+() const {
        return Rational(this->p, this->q);
    }

    friend const Rational operator+(const int, const Rational &);

    friend const Rational operator-(const int, const Rational &);

    friend const Rational operator*(const int, const Rational &);

    friend const Rational operator/(const int, const Rational &);

    friend const bool operator>(const int, const Rational &);

    friend const bool operator<(const int, const Rational &);

    friend const bool operator>=(const int, const Rational &);

    friend const bool operator<=(const int, const Rational &);

    friend const bool operator==(const int, const Rational &);

    friend const bool operator!=(const int, const Rational &);

    friend ostream &operator<<(ostream &, const Rational &);

    friend istream &operator>>(istream &, Rational &);
};

const Rational operator+(const int a, const Rational &rat) {
    return Rational(a) + rat;
}

const Rational operator-(const int a, const Rational &rat) {
    return Rational(a) - rat;
}

const Rational operator*(const int a, const Rational &rat) {
    return Rational(a) * rat;
}

const Rational operator/(const int a, const Rational &rat) {
    return Rational(a) / rat;
}

const bool operator>(const int a, const Rational &rat) {
    return rat < a;
}

const bool operator<(const int a, const Rational &rat) {
    return rat > a;
}

const bool operator>=(const int a, const Rational &rat) {
    return rat <= a;
}

const bool operator<=(const int a, const Rational &rat) {
    return rat >= a;
}

const bool operator==(const int a, const Rational &rat) {
    return rat == a;
}

const bool operator!=(const int a, const Rational &rat) {
    return rat != a;
}

istream &operator>>(istream &is, Rational &rat) {
    char str[30];
    is >> str;
    char *split_str = strtok(str, "/"); //разделяем по '/'
    rat.p = atoi(split_str);
    split_str = strtok(NULL, "/");
    if (split_str != NULL)
        rat.q = atoi(split_str);
    else
        rat.q = 1;
    rat.reduce();
    return is;
}

ostream &operator<<(ostream &os, const Rational &rat) {
    os << rat.p;
    if (rat.q != 1)
        os << "/" << rat.q;
    return os;
}

int main() {
    int m, n, p, q;
    cin >> m >> n >> p >> q;

    Matrix<int> A(m, n), B(p, q);
    cin >> A >> B;

    A = A;
    try {
        cout << A + B * 2 - m * A << endl;
        cout << (A -= B += A *= 2) << endl;
        cout << (((A -= B) += A) *= 2) << endl;
    }
    catch (const MatrixWrongSizeError &) {
        cout << "A and B are of different size." << endl;
    }
    B = A;
    cout << B << endl;

    Rational r;
    cin >> r;
    Matrix<Rational> C(m, n), D(p, q);
    cin >> C >> D;
    try {
        cout << C * D << endl;
        cout << (C *= D) << endl;
        cout << C << endl;
    }
    catch (const MatrixWrongSizeError &) {
        cout << "C and D have not appropriate sizes for multiplication." << endl;
    }
    cout << C.getTransposed() * (r * C) << endl;
    cout << C.transpose() << endl;
    cout << C << endl;

    SquareMatrix<Rational> S(m);
    cin >> S;

    SquareMatrix<Rational> P(S);
    const SquareMatrix<Rational> &rS = S;
    cout << rS.getSize() << ' ' << rS.getDeterminant() << ' ' << rS.getTrace() << endl;
    cout << (S = S) * (S + rS) << endl;
    cout << (S *= S) << endl;
    C.transpose();
    cout << rS * C << endl;
    cout << S << endl;
    S = P;
    cout << (Rational(1, 2) * S).getDeterminant() << endl;
    try {
        cout << rS(0, 0) << endl;
        (S(0, 0) *= 2) /= 2;
        cout << rS(0, m) << endl;
    }
    catch (const MatrixIndexError &) {
        cout << "Index out of range." << endl;
    }
    cout << rS.getTransposed() << endl;
    try {
        cout << rS.getInverse() << endl;
        cout << S.invert().getTransposed().getDeterminant() << endl;
        cout << S << endl;
    }
    catch (const MatrixIsDegenerateError &) {
        cout << "Cannot inverse S." << endl;
    }
    return 0;
}