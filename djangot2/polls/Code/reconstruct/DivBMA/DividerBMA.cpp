 
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <map>
#include <set>

#include <string>
#include <iterator>
#include <time.h>
#include <math.h>
#include <random>
#include <chrono>
#include <cstdint>
#include <climits>
#include <fstream>
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

#define DES_LEN 152
#define HISTO_LEN 1000
#define MAX 200
#define SAME 0
#define DEL 1
#define INS 2
#define SUB 3
#define NUMBER_OF_BP 4

int lookup[MAX][MAX];


string _correct_binary_indel(int n, int m, int a, string y);
string _correct_q_ary_indel(int n, int m, int a, int b, int q, string y);
string convert_y_to_alpha(string y);
string genRandomSeq(int len);
vector<int> compute_indices(char y1, string x);
int count_embeddings(string a, string b);
string maximum_likelihood_old(set<string> &candidates, const vector<string> &cluster);
string maximum_likelihood(set<string> &candidates, const vector<string> &cluster);
void createNoisyCluster(vector<string> &cluster, string original, int cluster_size, float del_prob);
string reconstruction_new(vector<string> &cluster, int des_len);
string reconstruction_new_alg4(vector<string> &cluster, int des_len);
void reomoveByLength(set<string> &candidates, int len);
void removeNonSuperseq_withLen(set<string> &candidates, vector<string> &cluster, int len);
void removeNonSuperseq(set<string> &candidates, vector<string> &cluster, int len);
int compute_syndrome_binary(int m, int a, string y);
string convert_y_to_alpha(string y);
int compute_syndrome_q_ary(int m, int a, int b, int q, string y, int &sec);
bool is_codeword(string &y, int q);
string decode_VT(string y, int q, int n, int m , int a, int b);

vector<string> LCS(string X, string Y, int m, int n);

//set<string> LCS(string X, string Y, int m, int n);
void LCSLength(string X, string Y, int m, int n);

int design_len=DES_LEN;
int total_dis_tests = 0;
float prob = 0.8;
int clus_size = 6;
int num_of_test = 100000;
int count=0;
int q=4;
int scs_size=0;
int scs_after_filter_size=0;
int scs_most_likely_size=0;
int most_likelihood_count=0;
int pairs_histo[HISTO_LEN]={0};
int three_histo[HISTO_LEN]={0};
int four_histo[HISTO_LEN]={0};
int SR_histo[DES_LEN]={0};
int SR_histo_BMA[DES_LEN]={0};
int SR_histo_MATIKA[DES_LEN]={0};
int SR_histo_D_R[DES_LEN]={0};
int SR_histo_karin[DES_LEN]={0};

double i_prob=0.01;
double d_prob=0.01;
double s_prob=0.01;

set<string> ML;
map<int, int> scs_sizes;
map<int, int> scs_sizes_count;
map<int, int> mst_like;
map<int, int> mst_like_count;
//int table3[4][4]={{0}};
//int table4[4][4]={{0}};
//int table_l_3={0,1,3,4}=;
//int table_r_3=;
double error_rate_by_length[DES_LEN+50]={0};
double count_by_length[DES_LEN+50]={0};
map < string, int > table_1_rev;
map < string, int > table_2_rev;
vector<int> systematic_positions_step_1;
int lucky = 0;
int two = 0;
int three = 0;
int four = 0;
int nothing = 0;
ofstream myfile;


char intToDNA(int num){
    switch(num){
        case 0 : return 'A';
        case 1 : return 'C';
        case 2 : return 'G';
        case 3 : return 'T';
    }
    return 'N';
}

int DNAtoInt(char c){
    if (c=='A')
        return 0;
    if (c=='C')
        return 1;
    if (c=='G')
        return 2;
    if (c=='T')
        return 3;
    return -1;
}

char letToBp(int i){
    switch(i){
    case 0:
        return 'A';
    case 1:
        return 'C';
    case 2:
        return 'G';
    case 3:
        return 'T';
    }
    return 'N';
}

int letToInt(char c){
    switch(c){
    case 'A':
        return 0;
    case 'C':
        return 1;
    case 'G':
        return 2;
    case 'T':
        return 3;
    }
    return 4;
}


unsigned int edit_distance(const std::string& s1, const std::string& s2)
{
    const std::size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    d[0][0] = 0;
    for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

    for(unsigned int i = 1; i <= len1; ++i)
        for(unsigned int j = 1; j <= len2; ++j)
                      // note that std::min({arg1, arg2, arg3}) works only in C++11,
                      // for C++98 use std::min(std::min(arg1, arg2), arg3)
                      d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
    return d[len1][len2];
}

struct LetterOps {
    std::string insert; // string to insert before letter
    std::string CDR; // Copy or Delete or Substitute letter
    std::string opParam; // Which letter to copy or which to delete or which to substitute with
};

/* this is a function that help us print the trace of two sequences */
string printTrace(string word1, string word2,vector< vector<int > > backtrace){
    int i1 = (word1.size()), i2 = (word2.size());

    string trace="";
    while ((i1 > 0 && i2 >-1) || (i2 > 0&& i1>-1)) {
        switch (backtrace[i1][i2]) {
        case(SAME): // same
            trace.insert(trace.begin(), '-');
            //stream << "-";
            i1 -= 1;
            i2 -= 1;
            break;
        case(SUB):
                    trace.insert(trace.begin(), 'S');
            i1 -= 1;
            i2 -= 1;
            break;
        case(INS):
                    trace.insert(trace.begin(), 'I');

            i2 -= 1;
            break;
        case(DEL):
                    trace.insert(trace.begin(), 'D');

            i1 -= 1;
            break;
        default:
                cout << "ERR " << i1 << " " << i2 << " " <<  backtrace[i1][i2] << endl;

                cout << endl << backtrace[i1][i2] << " " << i1 << " " << i2 << endl;

                for(int i=0; i<=word1.size(); i++){
                    for(int j=0; j<=word2.size(); j++){
                        cout << backtrace[i][j] << " " ;
                    }
                    cout << endl;
                }
                exit(0);

        }

    }

    return trace;
}

string edit_distance_errorVec(const std::string& s1, const std::string& s2)
{
    const std::size_t len1 = s1.size(), len2 = s2.size();

    std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    std::vector<std::vector<int>> b(len1 + 1, std::vector<int>(len2 + 1));

    b[0][0]=SAME;
    d[0][0] = 0;
    for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len1; ++i) b[i][0] = DEL;

    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;
    for(unsigned int i = 1; i <= len2; ++i) b[0][i] = INS;

    //cout << "befor  " << endl;
    for(unsigned int i = 1; i <= len1; ++i){
        for(unsigned int j = 1; j <= len2; ++j){
                      // note that std::min({arg1, arg2, arg3}) works only in C++11,
                      // for C++98 use std::min(std::min(arg1, arg2), arg3)
                     
            if(s1[i - 1] == s2[j - 1]){
                //cout << "sa " << i << " " << j <<endl;
                d[i][j]=d[i - 1][j - 1];
                b[i][j]=SAME;
            }
            else{
                if(d[i - 1][j] + 1<=d[i][j-1] + 1){
                    if(d[i - 1][j] + 1<=d[i - 1][j - 1]+1){
                        d[i][j]=d[i - 1][j]+1;
                        b[i][j]=DEL;//case of deletion
                        //cout << "de1" << i << " " << j <<endl;


                    }
                    else{
                        d[i][j]=d[i - 1][j-1]+1;
                        b[i][j]=SUB;//case of // case of substitution
                        //cout << "su1" << i << " " << j << " " << s1[i - 1] << " " <<  s2[j - 1] << endl;

                    }
                }
                else{
                    if(d[i][j-1] + 1<=d[i - 1][j - 1]+1){
                        d[i][j]=d[i][j-1]+1;
                        b[i][j]=INS;//case of insertion
                        //cout << "in" <<endl;


                    }
                    else{
                         d[i][j]=d[i - 1][j-1]+1;
                        b[i][j]=SUB;//case of // case of substitution
                        //cout << "su1" <<endl;

                    }
                }
            }
        }
    }

    string trace = printTrace(s1, s2, b);

    return trace;
}

/* only ins - del */
unsigned int levenstein_dis(const std::string& s1, const std::string& s2)
{
    const std::size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    d[0][0] = 0;
    for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

    for(unsigned int i = 1; i <= len1; ++i)
        for(unsigned int j = 1; j <= len2; ++j)
                      // note that std::min({arg1, arg2, arg3}) works only in C++11,
                      // for C++98 use std::min(std::min(arg1, arg2), arg3)
                      d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1000) });
    return d[len1][len2];
}


void reverseStr(string& str)
{
    int n = str.length();
  
    // Swap character starting from two
    // corners
    for (int i = 0; i < n / 2; i++)
        swap(str[i], str[n - i - 1]);
}




string recon_edit_dis(const vector<string>  &cluster, int des_len){
    vector<string> in_length;
    vector<string> shorter;
    vector<string> longer;
    
    for(string s: cluster){
        if(s.length()>des_len){
            longer.push_back(s);
            continue;
        }
        if(s.length()<des_len){
            shorter.push_back(s);
            continue;
        }
        in_length.push_back(s);
    }
    
    return "";
}


map<char, vector<int> > LettersMap(const vector<string>& strings, const vector<int>& index) {
	map<char, vector<int> > lettersMap;
	for (int i = 0; i < (int) strings.size(); i++) {
		if (index[i] == 0) {
			lettersMap['$'].push_back(i);
		}
		else {
			lettersMap[strings[i][index[i] - 1]].push_back(i);
		}
	}
	return lettersMap;
}



/* create a random string of a given length */
string genRandomSeq(int len, int q){
    //random_device random_device; // create object for seeding
    //mt19937 engine{random_device()}; // create engine and seed it
    unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 engine(sd);
    uniform_int_distribution<> dist(0, q-1); // create distribution for integers with [1; 9] range
    string res = "";
    for(int i=0; i<len; i++){
        int iSecret = dist(engine); // generate pseudo random int
        //char c=intToDNA(iSecret);
        char c = letToBp(iSecret);
        //cout << iSecret;
        //cout << c;
        res.push_back(c);
    }
    return res;
}

/* compute the indice that characters y1 appears in string x */
vector<int> compute_indices(char y1, string x){
    vector<int> indices;
    for(int i=0; i<x.length(); i++){
        if(x[i]==y1)
            indices.push_back(i);
    }
    return indices;
}



char generateRandomDNA(float prob){
    if(prob <=0.25){
        return 'A';
    }
    if(prob <= 0.5){
        return 'C';
    }
    if(prob <= 0.75){
        return 'G';
    }
    if(prob <= 1.0){
        return 'T';
    }
    return 'X';
}

void createNoisyCluster_edit(vector<string> &cluster, string original, int cluster_size, float del_prob, float sub_prob, float ins_prob){
    bool ins = false;
    bool del = false;
    bool sub = false;
    /* initialize random seed: */
    //srand (time(NULL));
    //random_device random_device; // create object for seeding
    //mt19937 engine{random_device()}; // create engine and seed it
    //uniform_int_distribution<> dist(1,100); // create distribution for integers with [1; 9] range
    unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 engine(sd);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double iSecret;
    float prob;
    for (int i =0; i < cluster_size ; i++){
        string new_string  ="";
        int count=0;
        for(int j=0 ; j<original.length(); j++ ){
            ins = false;
            del = false;
            sub = false;

            /* Insertion */
            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            //cout << "iS " << prob << endl;

            if(prob<=1-ins_prob){
                //new_string.push_back(original[j]);
            }else{
                ins=true;
                iSecret = dist(engine); // generate pseudo random number
                prob=iSecret;
                char c = generateRandomDNA(prob);
                new_string.push_back(c);
                //new_string.push_back(original[j]);
                //continue;
            }
            /* Deletion */
            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            //cout << "iS " << prob << endl;
            if(prob<=1-del_prob){
                //new_string.push_back(original[j]);
            }else{
                del=true;
                //cout << "del " << endl;
                count++;
                continue;
            }

            /* Sub */
            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            //cout << "iS " << prob << endl;

            if(prob<=1-sub_prob){
                //new_string.push_back(original[j]);
            }else{
                sub=true;
                iSecret = dist(engine); // generate pseudo random number
                prob=iSecret;
                char c = generateRandomDNA(prob);
                while(c == original[j]){
                    iSecret = dist(engine);
                    c = generateRandomDNA(iSecret);
                }
                new_string.push_back(c);
                //continue;
            }
            if(!del && !sub){
                new_string.push_back(original[j]);

            }
            
        }
        
        /* Insertion of last symbol */
        iSecret = dist(engine); // generate pseudo random number
        prob=iSecret;

        if(prob<=1-ins_prob){

        }else{
            ins=true;
            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            char c = generateRandomDNA(prob);
            new_string.push_back(c);
        }
        
        cluster.push_back(new_string);
        new_string = "";
    }
}

void createNoisyCluster_edit_equal(vector<string> &cluster, string original, int cluster_size, float error_prob){
    bool ins = false;
    bool del = false;
    bool sub = false;
    /* initialize random seed: */
    //srand (time(NULL));
    //random_device random_device; // create object for seeding
    //mt19937 engine{random_device()}; // create engine and seed it
    //uniform_int_distribution<> dist(1,100); // create distribution for integers with [1; 9] range
    unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 engine(sd);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double iSecret;
    float prob;
    for (int i =0; i < cluster_size ; i++){
        string new_string  ="";
        int count=0;
        for(int j=0 ; j<original.length(); j++ ){
            ins = false;
            del = false;
            sub = false;


            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            //cout << prob << endl;
            if(prob<=1-error_prob){
                // NO ERROR
            }else{
                /* ERROR */
                iSecret = dist(engine); // generate pseudo random number
                prob=iSecret;
                //cout << prob << endl;
                if(prob<=0.1/*(1.0/3.0)*/){
                    /*insertion*/
                    ins=true;
                    iSecret = dist(engine); // generate pseudo random number
                    prob=iSecret;
                    char c = generateRandomDNA(prob);
                    new_string.push_back(c);
                }
                else if(prob>0.1/*1.0/3*/ && prob<=0.2/*2.0/3*/){
                    /*deletion*/
                    del=true;
                    count++;
                    continue;
                }
                else{
                    /* Sub */

                    sub=true;
                    iSecret = dist(engine); // generate pseudo random number
                    prob=iSecret;
                    char c = generateRandomDNA(prob);
                    while(c == original[j]){
                        iSecret = dist(engine);
                        c = generateRandomDNA(iSecret);
                    }
                    new_string.push_back(c);
                    continue;
                }
            }
            if(!del && !sub){
                new_string.push_back(original[j]);
            }
            
        }
        
        /* Insertion of last symbol */
        iSecret = dist(engine); // generate pseudo random number
        prob=iSecret;

        if(prob<=1-error_prob){

        }else{
            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            if(prob<=(1/3)){
                /*insertion*/
                ins=true;
                iSecret = dist(engine); // generate pseudo random number
                prob=iSecret;
                char c = generateRandomDNA(prob);
                new_string.push_back(c);
            }
        }
        
        cluster.push_back(new_string);
        new_string = "";
    }
}


void createNoisyCluster(vector<string> &cluster, string original, int cluster_size, float del_prob){
    /* initialize random seed: */
    //srand (time(NULL));
    //random_device random_device; // create object for seeding
    //mt19937 engine{random_device()}; // create engine and seed it
    //uniform_int_distribution<> dist(1,100); // create distribution for integers with [1; 9] range
    unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 engine(sd);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double iSecret;
    float prob;
    for (int i =0; i < cluster_size ; i++){
        string new_string  ="";
        int count=0;
        for(int j=0 ; j<original.length(); j++ ){
            iSecret = dist(engine); // generate pseudo random number
            prob=iSecret;
            if(prob<=1-del_prob){
                new_string.push_back(original[j]);
            }else{
                count++;
                continue;
            }
        }
        cluster.push_back(new_string);
        new_string = "";
    }
}

// this function remove candidate out of set of candidates  by their length
void reomoveByLength(set<string> &candidates, int len){
    auto it = candidates.begin();
    while (it != candidates.end()){
        string candid = *it;
        
        if (candid.length() != len){
            it = candidates.erase(it);
        }
        else{
            it++;
        }
    }
    
    it = candidates.begin();
    while (it != candidates.end()){
        string candid = *it;
        
        if (candid.length() != len){
            cout << "prob" <<endl;
        }
        it++;
    }

}

// Returns true if str1[] is a subsequence of str2[]. m is
// length of str1 and n is length of str2
bool isSubSequence(string str1, string str2, int m, int n)
{
    int j = 0; // For index of str1 (or subsequence
    
    // Traverse str2 and str1, and compare current character
    // of str2 with first unmatched char of str1, if matched
    // then move ahead in str1
    for (int i=0; i<n&&j<m; i++)
        if (str1[j] == str2[i])
            j++;
    
    // If all characters of str1 were found in str2
    return (j==m);
}

// this function remove candidate out of set of candidates  by their length
void removeNonSuperseq_withLen(set<string> &candidates, vector<string> &cluster, int len){
    auto it = candidates.begin();
    bool del = false;
    while (it != candidates.end()){
        del = false;
        string candid = *it;
        for(string read: cluster){
            if(read.length() > candid.length())
                continue;
            if (!isSubSequence(read, candid, read.length(), candid.length())){
                it = candidates.erase(it);
                del = true;
                break;
            }
        }
        if(!del){
            it++;
        }
    }

}



// this function remove candidate out of set of candidates  by their length
void removeNonSuperseq(set<string> &candidates, vector<string> &cluster, int len){
    auto it = candidates.begin();
    bool del = false;
    while (it != candidates.end()){
        del = false;
        string candid = *it;
        for(string read: cluster){
            if (!isSubSequence(read, candid, read.length(), candid.length())){
                it = candidates.erase(it);
                del = true;
                break;
            }
        }
        if(!del){
            it++;
        }
    }

}


int find_index_of_short(const vector<string> &cluster){
    int i=0;
    int min_len = cluster[0].length();
    int min_ind=0;
    for(string str: cluster){
        if(str.length()<=min_len){
            min_ind=i;
            min_len=str.length();
        }
        i++;
    }
    return min_ind;
}

int find_index_of_long(const vector<string> &cluster){
    int i=0;
    int max_len = cluster[0].length();
    int max_ind=0;
    for(string str: cluster){
        if(str.length()>=max_len){
            max_len=i;
            
        }
        i++;
    }
    return max_len;
}

/* check if str is super sequence of all sequence in a given cluster*/
bool isSupSeq(string str, const vector<string> &cluster){
    for(string cp:cluster){
        if(!isSubSequence(cp, str, cp.length(), str.length()))
            return false;
    }
    return true;
}




/// BMA algo

void padCluster(vector<string> &cluster, int des_len){
    for(string str: cluster){
        int mis = des_len - str.length();
        while(mis>0){
            str.push_back('-');
            mis--;
        }
    }
}


void createPointers(vector<int> &pointers, int cluster_size){
    for(int i=0; i<cluster_size; i++){
        pointers.push_back(0);
    }
    
}

char BMAMajority(vector<string> &cluster, vector<int> &pointers){
    //cout << "set  size is " << sequences.size() << endl;
    int majority[5] = {0};
    int max = majority[0];
    int max_i =0;
    auto  it  =cluster.begin();
    int current_seq=0;
    while(it!=cluster.end()){
        string  seq = *it;
        if(seq.empty()){
            current_seq++;
            it++;
            continue;
        }
        if(seq[pointers[current_seq]]==0){
            current_seq++;
            it++;
            continue;
        }
        if((seq[pointers[current_seq]])-'0' <= -1 || (seq[pointers[current_seq]])-'0' >=4){
            cerr << "darlin " << " " << pointers[current_seq] << " " << seq[pointers[current_seq]] << endl;
            //throw MyException("INVALID DNA Char ");
            //cout << "ERROERERE  " << endl;
        }
        else{
            majority[(seq[pointers[current_seq]])-'0']++;
        }
        current_seq++;
        it++;
    }


    for(int i=0; i<4; i++){
        if(majority[i] > max) {
            max_i= i;
            max = majority[i];
        }
    }
    //cout  << intToDNA(max_i) << endl;
    return (max_i)+'0';
}

void updateLetter(vector<string> &cluster, vector<int> &pointers, string &res, char c){
    res.push_back(c);
    int current=0;
    for(string seq:cluster){
        if(seq[pointers[current]]==c){
            pointers[current]++;
        }
        current++;
    }
    return;
}


string BMA(vector<string> &cluster, int des_len, int &origSize){

    string res = "";
    vector<int> pointers;
    int cluster_size = cluster.size();

    padCluster(cluster, des_len);

    createPointers(pointers, cluster_size);
    for(int i=0; i<des_len-1; i++){
        int c=BMAMajority(cluster, pointers);
        //cout << c;
        updateLetter(cluster, pointers, res, c);
        //cout <<endl;
    }
    //cout << res << endl;
    return res;
}

int NCR(int n, int r)
{
    if (r == 0) return 1;

    /*
     Extra computation saving for large R,
     using property:
     N choose R = N choose (N-R)
    */
    if (r > n / 2) return NCR(n, n - r);

    long res = 1;

    for (int k = 1; k <= r; ++k)
    {
        res *= n - k + 1;
        res /= k;
    }

    return res;
}

char computerMajoOfNextSymbol(vector <string> &cluster, int positions[], int clus_size){
    // Step 1: Computing the majority of specific position
    int majo[256]={0};
    for(int i=0; i<clus_size; i++){
        char c=cluster[i][positions[i]+1];
        int c_i=(int)c;
        majo[c_i]++;
    }
    int max_i=0;
    int max=0;
    for(int i=0; i<256; i++){
        if(majo[i]>max){
            max=majo[i];
            max_i=i;
        }
    }
    return max_i;
}

char computeNextMajority(vector <string> &cluster, int positions[], int clus_size, int &position_lcs, string lcs){
    // Step 1: Computing the majority of specific position
    int majo[256]={0};
    for(int i=0; i<clus_size; i++){
        char c=cluster[i][positions[i]];
        if(c>='A' && c <='Z'){
            char c=lcs[c-'A'+22*(position_lcs/22)];
            position_lcs++;
        }
        int c_i=(int)c;
        majo[c_i]++;
    }
    int max_i=0;
    int max=0;
    for(int i=0; i<256; i++){
        if(majo[i]>max){
            max=majo[i];
            max_i=i;
        }
    }
    
    // Step 2: here we verify that there are no insertions
    char next_c=computerMajoOfNextSymbol(cluster, positions,clus_size);
    for(int i=0; i<clus_size; i++){
        if(cluster[i][positions[i]]==max_i){ // case of there was not "ERROR"
            positions[i]++;
        }
        else if(cluster[i][positions[i]]==next_c) { // case there was an "deletion"
            cluster[i].insert(positions[i]++,"_");
            //break;
        }
        else if(cluster[i][positions[i]+1]==next_c) { // case there was an "substitution"
            //break;
        }
        else if(cluster[i][positions[i]+1]==max_i) { // case there was an "insertion"
            //cout << "cluster "<<i<<" was : "<< cluster[i]  << endl;
            cluster[i].erase(positions[i],1);
            //cout << "cluster "<<i<<" was : "<< cluster[i]  << endl;
            //exit(0);
            //positions[i]++;
        }
    }
    return max_i;
}

void alignCluster(vector<string> &cluster, int clus_size, int position, char c, char next_c){
    for(int i=0; i<clus_size; i++){
        if(cluster[i][position]==c){
            continue;
        }
        else{
            if(position<cluster[i].length()-1 && cluster[i][position+1]==c){
                cluster[i][position] = 'I';
            }
            else{
                //position<cluster[i].length()-1 && cluster[i][position+1]==c
                cluster[i].insert(position++, "_");

            }
                    /*if(j<cluster[i].length()-1&&cluster[i][j+1] == lcs[lcs_index]){
                        cluster[i][j] = 'I';
                    }*/
                    
                /*
                if(j>position && j-position<2){
                    int x=position;
                    for(; x<cluster[i].length()&&j<cluster[i].length(); x++){
                        cluster[i][x]=cluster[i][j++];
                    }
                    cluster[i][x]='\0';
                    break;
                }
                if(j<position && j-position>-2){
                    while(j<position)
                        cluster[i].insert(j++, "_");
                }*/
                
        }
    }
}


void divideClusterByLength(vector<string> &clusters, vector<string> &shorter, vector<string> &longer,
        vector<string> &inLen, int targetLength){
    for(vector<string>::iterator it=clusters.begin(); it!=clusters.end(); it++){
        if((*it).size()==targetLength){
            inLen.push_back(*it);
            continue;
        }
        if((*it).size()<targetLength){
            shorter.push_back(*it);
            continue;
        }
        if((*it).size()>targetLength){
            longer.push_back(*it);
            continue;
        }
    }
}

void insertToMajoritySameLength(vector<string> &inTargetLen, vector< vector<int> > &majority_matrix){
    //int i=0;

    
    for(vector<string>::iterator it=inTargetLen.begin(); it!=inTargetLen.end(); it++){
        string currentString=*it;
        /*if(!is_DNA_seq(currentString)){
            continue;
        }*/
        int i=0;
        for(string::iterator strIt=currentString.begin(); strIt!=currentString.end();strIt++){
            switch(*strIt){
                case 'A':
                    majority_matrix[0][i]++;
                    break;
                case '0':
                    majority_matrix[0][i]++;
                    break;
                case 'C':
                    majority_matrix[1][i]++;
                    break;
                case '1':
                    majority_matrix[1][i]++;
                    break;
                case 'G':
                    majority_matrix[2][i]++;
                    break;
                case '2':
                    majority_matrix[2][i]++;
                    break;
                case 'T':
                    majority_matrix[3][i]++;
                    break;
                case '3':
                    majority_matrix[3][i]++;
                    break;
            }
            i++;
        }
    }
}



void computeMajority(vector< vector<int> > &majority_matrix, string &majority_sequence, int targetLength){
    /*for(int j=0; j<targetLength; j++){
        majority_matrix[4][j]=majority_matrix[A][j];
        majority_sequence[j]='A';
    }*/
    for(int j=0; j<targetLength; j++){
        for(int i=0; i<NUMBER_OF_BP; i++){
            if(majority_matrix[i][j]>majority_matrix[4][j]){
                majority_matrix[4][j]=majority_matrix[i][j];
                majority_sequence[j]=letToBp(i);

            }
        }

    }

}

void initMajorities(vector< vector<int> > &majority_matrix, string &majority_seq, int targetLength){
    int i;
    for(i=0; i<5; i++){
        for(int j=0; j<targetLength; j++){
            majority_matrix[i][j]=0;
        }
    }
    for(i=0; i<targetLength; i++){
        majority_seq.push_back('N');
    }
}



void updateForShort(string &short_seq, vector< vector<int> > &majority_matrix,
        string &majority_seq, int targetLength, int used_deletions){
    /*if(!is_DNA_seq(short_seq)){
        return;
    }*/
    int number_of_possible_deletion=targetLength-short_seq.size()-used_deletions;
    int i=0;
    for(string::iterator it=short_seq.begin(); it!=short_seq.end() && i<targetLength; it++){
        /* case 1 if we have the same as the majority */
        if(majority_seq[i]=='N'){
            majority_matrix[letToInt(*it)][i]++;
            majority_seq[i]=*it;
            i++;
            continue;
        }
        if(*it==majority_seq[i]){
            majority_matrix[letToInt(*it)][i]++;
        }
        else{
            /*if(i<majority_seq.size()-1 && *it==majority_seq[i+1]){
                used_deletions++;
                number_of_possible_deletion--;
                it--;
                i++;
                continue;
            }*/
            /* case of substitution if we see the next bp is good for us*/
            if(*(it+1)==majority_seq[i+1]&&*(it+2)==majority_seq[i+2]){
                majority_matrix[letToInt(*it)][i]++;
                if(majority_matrix[letToInt(*it)][i]>majority_matrix[4][i]){
                    majority_seq[i]=*it;
                }
                i++;
                continue;
            }
            /*case two if we are second best or have an advangate less than 2  */
            if(majority_matrix[letToInt(*it)][i]+3>majority_matrix[4][i]){
                majority_matrix[letToInt(*it)][i]++;
                if(majority_matrix[letToInt(*it)][i]>majority_matrix[4][i]){
                    majority_seq[i]=*it;
                }
                i++;
                continue;

            }
            else{
                /* case 3 we insert a deletion here and go the next option */
                if(number_of_possible_deletion>0){
                    used_deletions++;
                    number_of_possible_deletion--;
                    it--; // it is decremeted by 1 since we want to stay in the same letter for next iteration
                }

            }
        }
    i++;
    }
}

void updateHistogramForShort(vector<string> &shorts, vector< vector<int> > &majority_matrix,
        string &majority_seq, int targetLength){
    for(vector<string>::iterator it=shorts.begin(); it!=shorts.end(); it++){
        updateForShort(*it, majority_matrix, majority_seq, targetLength, 0);
    }
    return;
}

void updateHistForLong(string &long_seq, vector< vector<int> > &majority_matrix,
        string &majority_seq, int targetLength, int used_insertions){
    /*if(!is_DNA_seq(long_seq)){
        return;
    }*/
    int i=0;
    int number_of_possible_insertions=-long_seq.size()-targetLength-used_insertions;
    for(string::iterator it=long_seq.begin(); it!=long_seq.end() && i<targetLength; it++){
        if(majority_seq[i]=='N'){
            majority_matrix[letToInt(*it)][i]++;
            majority_seq[i]=*it;
            i++;
            continue;
        }
        if(*it==majority_seq[i]){
            majority_matrix.at(letToInt(*it)).at(i)++;
        }
        else{
            /*case two if we are second best or have an advangate less than 2  */
            /* if we see that the next char is not good for us */
            if(    !(*(it+1)==majority_seq[i]||*(it+2)==majority_seq[i]) &&majority_matrix[letToInt(*it)][i]+3>majority_matrix[4][i]
                ){
                majority_matrix[letToInt(*it)][i]++;
                if(majority_matrix[letToInt(*it)][i]>majority_matrix[4][i]){
                    majority_seq[i]=*it;
                }
                i++;
                continue;
            }
            else{
                /* case 3 we insert a deletion here and go the next option */
                if(number_of_possible_insertions>0){
                    used_insertions++;
                    number_of_possible_insertions--;
                    i--; // i is decremented by 1 in order to perform insertion.
                }
            }
        }
    
    i++;
    }
}

void updateHistogramForLong(vector<string> &longs, vector< vector<int> > &majority_matrix,
        string &majority_seq, int targetLength){
    for(vector<string>::iterator it=longs.begin(); it!=longs.end(); it++){
        updateHistForLong(*it, majority_matrix, majority_seq, targetLength, 0);
    }
    return;
}

/* this function compute the majority strands including the insertion and deletion */
string makeMajorityEditDistance(vector<string> &clusters, int targetLength){
    /* here we create and initialize majority matrix
     * the 4 first rows are for the BP and the last one is the maximum
     * an additional string represents the majority so far;
     */
    string majority_seq;
    vector< vector<int> > majority_matrix(NUMBER_OF_BP+1, vector<int>(targetLength));
    initMajorities(majority_matrix, majority_seq, targetLength);
            //cerr << "Init Completed" << endl;

    /* phase 1 - divide our cluster to string with shorter, longer and exact length of
     * the target length
     */
    vector<string> shorter, longer, inTargetLen;
    divideClusterByLength(clusters, shorter, longer, inTargetLen, targetLength);
                //cerr << "divide Completed" << endl;

    /*phase 2 - filing the matrix according to the strands with exactly the same length */
    insertToMajoritySameLength(inTargetLen, majority_matrix);
                    //cerr << "Insert Same Completed" << endl;

    /* computing the majority so far */
    computeMajority(majority_matrix, majority_seq, targetLength);
                    //cerr << "Compute Maj Completed" << endl;

    /* computing majority for shorts: aka we need insertion */
    updateHistogramForShort(shorter, majority_matrix, majority_seq, targetLength);
                    //cerr << "Update Shorter Completed" << endl;

    /* computing majority for long aka we need deletions */
    updateHistogramForLong(longer,  majority_matrix, majority_seq, targetLength);
                    //cerr << "Update Longer Completed" << endl;

    return majority_seq;
}

void prediction(map< string, vector<string> > &clusters){
    for(map<string, vector<string> >::iterator it=clusters.begin(); it!=clusters.end(); it++){
        cerr << it->first << endl;
        //cout << it->first << endl;
        string input = it->first;
        cout << input << endl;
        string prediction=makeMajorityEditDistance(it->second, it->first.size());
        cout << prediction << endl;
        //printPrediction(prediction, input);
    }
}


int get_max(vector<int> &ptrs){
    int max_i=ptrs[0];
    for(int p:ptrs){
        if(p>max_i){
            max_i=p;
        }
    }
    return max_i;
}





char getMajorityByPointers(vector<string> &cluster, vector<int> &ptrs){
    int cl_size=cluster.size();
    int majo[NUMBER_OF_BP] = {0,0,0,0};
    for(int i=0; i<cl_size; i++){
        if(ptrs[i] < cluster[i].length())
            majo[ letToInt(cluster[i][ ptrs[i] ]) ]++;
    }
    int max_i=0;
    int max_value=majo[max_i];
    for(int i=1 ; i<NUMBER_OF_BP; i++){
        if(majo[i]>max_value){
            max_i=i;
            max_value=majo[i];
        }
    }
    
    return letToBp(max_i);
}



string LOOKBack_Majority(vector<string> &cluster, int des_len){
    string output="";
    char base;
    char prev_base='N';
    vector<int> ptrs;
    for(int i=0; i < cluster.size(); i++){
        ptrs.push_back(0);
    }
    for(int i=0; i<des_len; i++){
        if(get_max(ptrs)>des_len){
            break;
        }
        base = getMajorityByPointers(cluster, ptrs);
        output.push_back(base);
        for(int j=0; j < cluster.size(); j++){
            if(prev_base!='N' && cluster[j][ptrs[j]]!=prev_base && ptrs[j] < i){
                if(j < ptrs.size() - 2){
                    if(cluster[j][ptrs[j+1]] == prev_base && cluster[j][ptrs[j+2]]== base){
                        /* case of insertion*/
                        if(ptrs[j] < ( (cluster[j]).length()-2 )  ){
                            ptrs[j]+=2;
                        }
                        else{
                            ptrs[j] = ptrs[j];
                        }
                        
                    }
                    else if(cluster[j][ptrs[j+1]]==base){
                        /* case of insertion*/
                        if(ptrs[j] < ( (cluster[j]).length()-1 )  ){
                            ptrs[j]+=1;
                        }
                        else{
                            ptrs[j] = ptrs[j];
                        }
                    }
                }
            }
            if(ptrs[j] < (cluster[j]).length()){
                char current_base = cluster[j][ptrs[j]];
                if(current_base==base){
                    ptrs[j] += 1;

                }
                else{
                    ptrs[j] = ptrs[j];
                }
            }
            
        }
        prev_base=base;
    }
    return output;
}


char getMajorityByPointers_window(vector<string> &cluster, vector<int> &ptrs, int w, vector<int> variant_reads){
    int cl_size=cluster.size();
    int majo[NUMBER_OF_BP+1] = {0,0,0,0,0}; // the last position refer to the case of an emptystring
    for(int i=0; i<cl_size; i++){
        if(find(variant_reads.begin(), variant_reads.end(),i)!=variant_reads.end()){
            continue;
        }
        if(ptrs[i]+w<cluster[i].length()){
            majo[ letToInt(cluster[i][ ptrs[i]+w ]) ]++;
        }
        else{
            continue;
        }
    }
    int max_i=0;
    int max_value=majo[max_i];
    for(int i=1 ; i<NUMBER_OF_BP; i++){
        if(majo[i]>max_value){
            max_i=i;
            max_value=majo[i];
        }
    }
    if(max_i<NUMBER_OF_BP)
        return letToBp(max_i);
    else
        return 0;
}

string majority_window(vector<string> &cluster, int des_len, int w, vector<int> &ptrs, vector<int> variant_reads){
    string window_maj = "";
    for(int i=0; i<=w; i++){
        char c = getMajorityByPointers_window(cluster, ptrs, i, variant_reads);
        window_maj.push_back(c);
        if (c==0)
            break;
    }
    return window_maj;
}

bool check_Substitution(string current_window, string window_maj, int window_size){
    return current_window.compare(1,window_size-1,window_maj,1,window_size-1)==0;
}

bool check_Deletion(string current_window, string window_maj, int window_size){
    return current_window.compare(0,window_size-1,window_maj,1,window_size-1)==0;
}

bool check_Insertion(string current_window, string window_maj, int window_size){
    return current_window.compare(1,window_size-1,window_maj,0,window_size-1)==0;
}

string LOOKAhead_Majority(vector<string> &cluster, int des_len, int w){
    string output="";
    char base;
    char prev_base='N';
    vector<int> ptrs;
    for(int i=0; i < cluster.size(); i++){
        ptrs.push_back(0);
    }
    for(int i=0; i<des_len; i++){
        if(get_max(ptrs)>des_len){
            break;
        }
        base = getMajorityByPointers(cluster, ptrs);
        vector<int> variant_reads;
        for(int j=0; j<cluster.size(); j++){
            if(base!=cluster[j][ptrs[j]]){
                variant_reads.push_back(j);
            }
        }
        output.push_back(base);
        string window_maj = majority_window(cluster, des_len, w, ptrs, variant_reads);
        int new_w = window_maj.length();
        for(int j=0; j < cluster.size(); j++){
            if(ptrs[j]>=cluster[j].length() ){
                continue;
            }
            if(cluster[j][ptrs[j]]==base){
                ptrs[j] += 1;
                continue; // Case #1: No error in trace j
            }
            else{
                string current_window;
                int k=0;
                for(k=0; k<=w && cluster[j][ptrs[j]+k]!=0; k++){
                    current_window.push_back(cluster[j][ptrs[j]+k]);
                }
                int max_w = min(k,new_w);
                if(max_w<=1){
                    ptrs[j] += 1;
                    continue;
                }
                /* case of substitution */
                if(current_window.compare(1,max_w-1,window_maj,1,max_w-1)==0){
                    ptrs[j] += 1;
                    continue;
                }
                /* case of insertion */
                if(current_window.compare(1,max_w-1,window_maj,0,max_w-1)==0){
                    ptrs[j] += 2;
                    continue;
                }

                /* case of deletion */
                if(current_window.compare(0,max_w-1,window_maj,1,max_w-1)==0){
                    ptrs[j] += 0;
                    continue;
                }

                
                // Second Check by Edit Distance.
                /* case of substitution */
                if(edit_distance(current_window.substr(1,max_w-1), window_maj.substr(1, max_w-1))<2){
                    ptrs[j] += 1;
                    continue;
                }
                /* case of insertion */
                if(edit_distance(current_window.substr(1,max_w-1), window_maj.substr(0, max_w-1))<2){
                    ptrs[j] += 2;
                    continue;
                }

                
                /* case of deletion */
                if(edit_distance(current_window.substr(0,max_w-1), window_maj.substr(1, max_w-1))<2){
                    ptrs[j] += 0;
                    continue;
                }
                

                
                ptrs[j]+=1;
                
            }
        }
        prev_base=base;
    }
    return output;
}

int getTestNum(const string& inputFilename){
    ifstream input;
    input.open(inputFilename.c_str());
    if (!input.is_open()) {
        cout << "Failed opening input file!" << endl;
        return -1;
    }
    string line;
    int count=0;
    while (getline(input, line)) {
        if (line[0]=='*') {
            count++;
        }
    }
    return count;
}

int EditDistanceArray(const string& X, const string& Y, int m, int n, vector<vector<int> >& dp) {
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0) {
                dp[i][j] = j;
            }
            else if (j == 0) {
                dp[i][j] = i;
            }
            else if (X[i - 1] == Y[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else {
                dp[i][j] = 1 + min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]);
            }
        }
    }
    return dp[m][n];
}


// subPriority, delPriority, insPriority - permutation of 1,2,3
vector<LetterOps> BacktrackEditDistancePriority(const string& X, const string& Y, int m, int n,
        const vector<vector<int> >& dp, const int subPriority, const int insPriority, const int delPriority) {
    vector<LetterOps> s(X.size() + 1);
    if (m == 0) {
        s[0].insert = Y.substr(0, n);
        return s;
    }
    else if (n == 0) {
        for (int index = 0; index < m; index++) {
            s[index].CDR = "D";
        }
        return s;
    }
    if (X[m - 1] == Y[n - 1]) {
        vector<LetterOps> s = BacktrackEditDistancePriority(X, Y, m - 1, n - 1, dp, subPriority, insPriority,
                delPriority);
        s[m - 1].CDR = "C";
        s[m - 1].opParam = string(1, X[m - 1]);
        return s;
    }
    else {
        int subEqual = 0, delEqual = 0, insEqual = 0;
        if (dp[m][n] == dp[m - 1][n - 1] + 1) {
            subEqual = 1;
        }
        if (dp[m][n] == dp[m][n - 1] + 1) {
            insEqual = 1;
        }
        if (dp[m][n] == dp[m - 1][n] + 1) {
            delEqual = 1;
        }
        int subScore = subEqual * subPriority, delScore = delEqual * delPriority, insScore = insEqual * insPriority;
        bool chooseSub = false, chooseDel = false, chooseIns = false;
        if (subScore > delScore and subScore > insScore) {
            chooseSub = true;
        }
        else if (delScore > subScore and delScore > insScore) {
            chooseDel = true;
        }
        else {
            assert(insScore > subScore and insScore > delScore);
            chooseIns = true;
        }
        if (chooseSub) {      // replace
            vector<LetterOps> s1 = BacktrackEditDistancePriority(X, Y, m - 1, n - 1, dp, subPriority, insPriority,
                    delPriority);
            s1[m - 1].CDR = "R";
            s1[m - 1].opParam = string(1, Y[n - 1]);
            return s1;
        }

        else if (chooseIns) {         // insert
            vector<LetterOps> s2 = BacktrackEditDistancePriority(X, Y, m, n - 1, dp, subPriority, insPriority,
                    delPriority);
            s2[m].insert += Y[n - 1];
            return s2;
        }
        else {            // delete
            assert(chooseDel);
            vector<LetterOps> s3 = BacktrackEditDistancePriority(X, Y, m - 1, n, dp, subPriority, insPriority,
                    delPriority);
            s3[m - 1].CDR = "D";
            return s3;
        }
    }
}

vector<LetterOps> BacktrackEditDistanceRandom(const string& X, const string& Y, int m, int n,
        const vector<vector<int> >& dp, vector<int>& priorities, mt19937& generator) {
    vector<LetterOps> s(X.size() + 1);
    if (m == 0) {
        s[0].insert = Y.substr(0, n);
        return s;
    }
    else if (n == 0) {
        for (int index = 0; index < m; index++) {
            s[index].CDR = "D";
        }
        return s;
    }
    if (X[m - 1] == Y[n - 1]) {
        vector<LetterOps> s = BacktrackEditDistanceRandom(X, Y, m - 1, n - 1, dp, priorities, generator);
        s[m - 1].CDR = "C";
        s[m - 1].opParam = string(1, X[m - 1]);
        return s;
    }
    else {
        //random_shuffle(priorities.begin(),priorities.end());
        shuffle(priorities.begin(), priorities.end(), generator);
        int subEqual = 0, delEqual = 0, insEqual = 0;
        if (dp[m][n] == dp[m - 1][n - 1] + 1) {
            subEqual = 1;
        }
        if (dp[m][n] == dp[m][n - 1] + 1) {
            insEqual = 1;
        }
        if (dp[m][n] == dp[m - 1][n] + 1) {
            delEqual = 1;
        }
        int subScore = subEqual * priorities[0], delScore = delEqual * priorities[2], insScore = insEqual
                * priorities[1];
        bool chooseSub = false, chooseDel = false, chooseIns = false;
        if (subScore > delScore and subScore > insScore) {
            chooseSub = true;
        }
        else if (delScore > subScore and delScore > insScore) {
            chooseDel = true;
        }
        else {
            assert(insScore > subScore and insScore > delScore);
            chooseIns = true;
        }
        if (chooseSub) {      // replace
            vector<LetterOps> s1 = BacktrackEditDistanceRandom(X, Y, m - 1, n - 1, dp, priorities, generator);
            s1[m - 1].CDR = "R";
            s1[m - 1].opParam = string(1, Y[n - 1]);
            return s1;
        }

        else if (chooseIns) {         // insert
            vector<LetterOps> s2 = BacktrackEditDistanceRandom(X, Y, m, n - 1, dp, priorities, generator);
            s2[m].insert += Y[n - 1];
            return s2;
        }
        else {            // delete
            assert(chooseDel);
            vector<LetterOps> s3 = BacktrackEditDistanceRandom(X, Y, m - 1, n, dp, priorities, generator);
            s3[m - 1].CDR = "D";
            return s3;
        }
    }
}

// priority: R - replace, I - insert, D - delete
// 0 - RID = [3,2,1]
// 1 - RDI = [3,1,2]
// 2 - IRD = [2,3,1]
// 3 - IDR = [1,3,2]
// 4 - DRI = [2,1,3]
// 5 - DIR = [1,2,3]
// 6 - Random branch each time
vector<LetterOps> ComputeEditDistancePriority(const string& X, const string& Y, const int priority,
        mt19937& generator) {

    int m = X.length(), n = Y.length();
    vector<vector<int> > dp(m + 1, vector<int>(n + 1));
    vector<LetterOps> result;
    EditDistanceArray(X, Y, m, n, dp);
    if (priority == 0) {
        result = BacktrackEditDistancePriority(X, Y, m, n, dp, 3, 2, 1);
    }
    else if (priority == 1) {
        result = BacktrackEditDistancePriority(X, Y, m, n, dp, 3, 1, 2);
    }
    else if (priority == 2) {
        result = BacktrackEditDistancePriority(X, Y, m, n, dp, 2, 3, 1);
    }
    else if (priority == 3) {
        result = BacktrackEditDistancePriority(X, Y, m, n, dp, 1, 3, 2);
    }
    else if (priority == 4) {
        result = BacktrackEditDistancePriority(X, Y, m, n, dp, 2, 1, 3);
    }
    else if (priority == 5) {
        result = BacktrackEditDistancePriority(X, Y, m, n, dp, 1, 2, 3);
    }
    else {
        assert(priority == 6);
        vector<int> priorities { 1, 2, 3 };
        result = BacktrackEditDistanceRandom(X, Y, m, n, dp, priorities, generator);
    }
    return result;
}


//vector<LetterOps> ComputeEditDistanceOperations(const string& X, const string& Y) {
//
//    int m = X.length(), n = Y.length();
//    vector<vector<int> > dp(m + 1, vector<int>(n + 1));
//
//    EditDistanceArray(X, Y, m, n, dp);
//    return BacktrackEditDistanceOperations(X, Y, m, n, dp);
//}

int ComputeEditDistanceNum(const string& X, const string& Y) {

    int m = X.length(), n = Y.length();
    vector<vector<int> > dp(m + 1, vector<int>(n + 1));

    return EditDistanceArray(X, Y, m, n, dp);
}

map<string, double> CountOperations(const vector<LetterOps>& opList) {
    map<string, double> count;
    count["C"] = 0;
    count["I"] = 0;
    count["D"] = 0;
    count["R"] = 0;

    for (vector<LetterOps>::const_iterator it = opList.begin(); it != opList.end(); it++) {
        if (not it->CDR.empty()) {
            count[it->CDR]++;
        }
        count["I"] += it->insert.size();
    }
    return count;
}

// main function
int main(int argc, char *argv[])
{
    if(argc<3){
        cout<< "not enough argument" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    string output_path = argv[2];
    auto start = high_resolution_clock::now();

    double succes_rate=0.0;
    int testNum=getTestNum(input_file);
    try{
        int counterOfGoodLen=0;
        string original;
        vector<string> cluster;
        

        
        double res_del;
        double res_ins;
        double res_sub;
        
    for(d_prob = 0.01 ; d_prob<=0.01; d_prob+=0.01){
        unsigned sd = chrono::high_resolution_clock::now().time_since_epoch().count();
        mt19937 generator(sd);
        int roundFinalGuessEditDist = 0;
        double cumTotalFinalGuessEditDist = 0;
        double cumFinalGuessSubstitutions = 0, cumFinalGuessInsertions = 0, cumFinalGuessDeletions = 0;
        double error_rate=0.0;
        map<int, int> editDistanceHist;
            cluster.clear();
            std::ifstream strands_file;
            strands_file.open(input_file);
        strands_file.close();
        strands_file.open(input_file);
            std::ofstream results_fail;
            results_fail.open(output_path+"/output-results-fail.txt");
            std::ofstream output;
            output.open(output_path+"/output.txt");
            std::ofstream results_success;
            results_success.open(output_path+"/output-results-success.txt");
            //cerr << "file" << endl;
            if (!strands_file) {
                    return -1;
            }
            int nt=0;
            int ct=0;
        int i=0;
            while(!strands_file.eof()){
                cout << int(100*(double(++i)/testNum)) << endl;


                cluster.clear();
                if(strands_file.eof()){
                    num_of_test=nt;
                    break;
                }
                string original_bck=original;
                getline(strands_file, original);
                if(original.length()<5){
                    original=original_bck;
                }
                if(strands_file.eof()){
                    num_of_test=nt;
                    break;
                }
                string line="";
                getline(strands_file, line); // line of "****"
                //int nccc=0;
                while(line.length()>1 && !strands_file.eof()){
                    getline(strands_file, line);
                    //if(nccc++ < 25)
                        cluster.push_back(line);
                }
                getline(strands_file, line);
                vector <string> reverse_cluster;
                for(string cp:cluster){
                    reverseStr(cp);
                    reverse_cluster.push_back(cp);
                }
                string matika = makeMajorityEditDistance(cluster, original.length());
                string matika_rev =makeMajorityEditDistance(reverse_cluster, original.length());
                string nmatika="";
                for(int k=0; k<original.length(); k++){
                    if(k<original.length()/2)
                        nmatika.insert(nmatika.begin(), matika_rev[k]);
                }
                for(int k=(original.length()+1)/2-1; k>=0; k--){
                    nmatika.insert(nmatika.begin(), matika[k]);

                }
                matika = nmatika;

                
                string recon=matika;
                //int dis = int(edit_distance(recon, original));
                //for (int i=dis; i<DES_LEN; i++){
                //    SR_histo_MATIKA[i]++;
                //}
                //total_dis_tests_Matika += dis;
                //succes_rate_Matika+= dis > 0 ? 0 : 1;
                roundFinalGuessEditDist = ComputeEditDistanceNum(original, recon);
                cout << recon << endl;
                editDistanceHist[roundFinalGuessEditDist]++;
                vector<LetterOps> result = ComputeEditDistancePriority(recon, original, 0, generator);
                map<string, double> countOperations = CountOperations(result);
                assert(countOperations["I"] + countOperations["D"] + countOperations["R"] == roundFinalGuessEditDist);
                if(original.length()){
                    cumTotalFinalGuessEditDist =((i-1)*cumTotalFinalGuessEditDist+double(roundFinalGuessEditDist)/(original.length() ))/i;
                    cumFinalGuessSubstitutions =((i-1)*cumFinalGuessSubstitutions+(countOperations["R"])/(original.length() ))/i;
                    cumFinalGuessInsertions =((i-1)*cumFinalGuessInsertions+(countOperations["I"])/(original.length() ))/i;
                    cumFinalGuessDeletions =((i-1)*cumFinalGuessDeletions+((countOperations["D"])/(original.length())))/i;
                    error_rate= ((i-1) *error_rate +(double(roundFinalGuessEditDist)/(original.length())))/i;
                }
                
                if(roundFinalGuessEditDist>0){
                    results_fail << "Cluster Num: " << ct << endl;
                    results_fail << original << endl;
                    results_fail << recon << endl;
                    results_fail << "Distance: " << roundFinalGuessEditDist << endl << endl;
                }
                else{
                    results_success << "Cluster Num: " << ct << endl;
                    results_success << original << endl;
                    results_success << recon << endl;
                    results_success << "Distance: " << roundFinalGuessEditDist << endl << endl;
                }
            
                nt++;
                ct++;
                num_of_test=nt;
            }
        map<int, int>::reverse_iterator rit = editDistanceHist.rbegin(); // points to last element in map
        int highestED = rit->first;
        int cumDist = 0;
        cout << "Total number of clusters: " << testNum << endl;
        output << "Total number of clusters: " << testNum << endl;

        cout << "Edit distance hist:" << endl;
        output << "Edit distance hist:" << endl;
        for (int i = 0; i <= highestED; i++) {
            cumDist += editDistanceHist[i];
            cout << i << "\t" << cumDist << endl;
            output << i << "\t" << cumDist << endl;
        }

        cout << "Substitution rate:\t" << cumFinalGuessSubstitutions << endl;
        cout << "Deletion rate:\t" << cumFinalGuessDeletions << endl;
        cout << "Insertion rate:\t" << cumFinalGuessInsertions << endl;
        //cout << "guess edit dist:\t" << cumTotalFinalGuessEditDist << endl;
        cout << "Error rate:\t" << error_rate << endl;
        cout << "Success rate:\t" << (double)(editDistanceHist[0])/testNum << endl;
        //cout << "number of majority test " << majoCounter << endl;

        output << "Substitution rate:\t" << cumFinalGuessSubstitutions << endl;
        output << "Deletion rate:\t" << cumFinalGuessDeletions << endl;
        output << "Insertion rate:\t" << cumFinalGuessInsertions << endl;
        //output << "guess edit dist:\t" << cumTotalFinalGuessEditDist << endl;
        output << "Error rate:\t" << error_rate << endl;
        output << "Success rate:\t" << (double)(editDistanceHist[0])/testNum << endl;
        //output << "number of majority test " << majoCounter << endl;
        //input.close();
        //output.close();
    }

    }
    catch(exception& e){
        cout<<e.what()<<endl;

        cerr<<e.what()<<endl;

    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    cout << endl << endl << "Time taken by function: "
        << duration.count() << " microseconds" << endl;
    //exit(0);
    return 0;
}
