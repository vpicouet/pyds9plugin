#set stuff {}
#proc MyMoveCB {varname id} {
#upvar #0 $varname var
#global $varname
#puts "hello world"
#puts "the varname passed is $varname"
#puts "and x is $id"
#puts "and id is $id"
#}
#

global stuff
#array set stuff {}
set stuff(1) aaa
set stuff(2) bbb
set stuff(3) ccc

proc MyMoveCB {varname id} {
   upvar #0 $varname var
   global $varname

   puts "the varname passed is $varname"
   puts "and its value(1) is $var(1)"
   puts "and its value(2) is $var(2)"
   puts "and its value(3) is $var(3)"
   puts "and id is $id"
}
