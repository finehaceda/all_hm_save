#!/usr/bin/perl
use warnings;
use Getopt::Long;

@input = ();
$output = "";

GetOptions('i=s' => \@input, 'o=s' => \$output);

open(W_MYFILE, ">$output");

for($i = 0; $i < @input; $i++)
{
	open(MYFILE, "<$input[$i]");
	$size = 0;
	while($line = <MYFILE>)
	{
		print W_MYFILE ($line);
		$size++;
	}
	print W_MYFILE ("\n");
	print("$size\n");
	close(MYFILE);
}

close(W_MYFILE);
