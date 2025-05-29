#!/usr/bin/perl
use warnings;
use Getopt::Long;

$input1 = "";
$input2 = "";
$output1 = "";
$output2 = "";
$output3 = "";
$output4 = "";
$output5 = "";
$output6 = "";

GetOptions('i1=s' => \$input1, 'i2=s' => \$input2, 'o1=s' => \$output1, 'o2=s' => \$output2, 'o3=s' => \$output3, 'o4=s' => \$output4, 'o5=s' => \$output5, 'o6=s' => \$output6);

open(W_MYFILE1, ">$output1");
open(W_MYFILE2, ">$output2");
open(W_MYFILE3, ">$output3");
open(W_MYFILE4, ">$output4");
open(W_MYFILE5, ">$output5");
open(W_MYFILE6, ">$output6");

open(MYFILE1, "<$input1");
open(MYFILE2, "<$input2");

$line = <MYFILE1>;
@array = split(/\t|\n|\s+/, $line);
for($i = 0; $i < $array[0] - 1; $i++)
{
	$line = <MYFILE2>;
	print W_MYFILE1 ($line);
}
$line = <MYFILE2>;
$a = substr($line, 0, length($line) - 1);
print W_MYFILE1 ($a);

$line = <MYFILE1>;
@array = split(/\t|\n|\s+/, $line);
for($i = 0; $i < $array[0] - 1; $i++)
{
        $line = <MYFILE2>;
        print W_MYFILE2 ($line);
}
$line = <MYFILE2>;
$a = substr($line, 0, length($line) - 1);
print W_MYFILE2 ($a);

$line = <MYFILE1>;
@array = split(/\t|\n|\s+/, $line);
for($i = 0; $i < $array[0] - 1; $i++)
{
        $line = <MYFILE2>;
        print W_MYFILE3 ($line);
}
$line = <MYFILE2>;
$a = substr($line, 0, length($line) - 1);
print W_MYFILE3 ($a);

$line = <MYFILE1>;
@array = split(/\t|\n|\s+/, $line);
for($i = 0; $i < $array[0] - 1; $i++)
{
        $line = <MYFILE2>;
        print W_MYFILE4 ($line);
}
$line = <MYFILE2>;
$a = substr($line, 0, length($line) - 1);
print W_MYFILE4 ($a);

$line = <MYFILE1>;
@array = split(/\t|\n|\s+/, $line);
for($i = 0; $i < $array[0] - 1; $i++)
{
        $line = <MYFILE2>;
        print W_MYFILE5 ($line);
}
$line = <MYFILE2>;
$a = substr($line, 0, length($line) - 1);
print W_MYFILE5 ($a);

$line = <MYFILE1>;
@array = split(/\t|\n|\s+/, $line);
for($i = 0; $i < $array[0] - 1; $i++)
{
        $line = <MYFILE2>;
        print W_MYFILE6 ($line);
}
$line = <MYFILE2>;
$a = substr($line, 0, length($line) - 1);
print W_MYFILE6 ($a);

close(MYFILE1);
close(MYFILE2);

close(W_MYFILE1);
close(W_MYFILE2);
close(W_MYFILE3);
close(W_MYFILE4);
close(W_MYFILE5);
close(W_MYFILE6);
