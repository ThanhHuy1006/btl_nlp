#2.1
import os
output_dir = os.path.join("..", 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Định nghĩa ngữ pháp
grammar = """
  S -> POSITIVE | NEGATIVE
  POSITIVE -> REQUEST_MORE_INFO | EXPRESS_INTEREST | ASK_DETAIL
  NEGATIVE -> EXCUSE | REJECT_OFFER | STATE_UNINTEREST
  REQUEST_MORE_INFO -> "tell" "me" "more" | "can" "you" "give" "me" "more" "details"
  EXPRESS_INTEREST -> "I'm" "interested" "in" "this" "product" | "this" "sounds" "interesting"
  ASK_DETAIL -> "can" "this" "product" "help" "me" "save" "costs"
  EXCUSE -> "I'm" "not" "free" "at" "the" "moment" | "I" "can't" "talk" "right" "now"
  REJECT_OFFER -> "no" "thanks" | "I'm" "not" "interested" "in" "it"
  STATE_UNINTEREST -> "I" "don't" "buy" "things" "from" "unsolicited" "calls"
"""

# Lưu ngữ pháp vào file
with open( os.path.join(output_dir, 'grammar.txt'), 'w') as file:
    file.write(grammar)
