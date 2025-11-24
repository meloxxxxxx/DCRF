import json
import re
from z3 import *

class LogicParser:
    """
    A class to parse first-order logic text and convert it to Z3 expressions.
    """

    def __init__(self):
        self.Entity = DeclareSort('Entity')
        self.predicates = {}
        self.constants = {}
        # Pre-compile regex for efficiency
        self.term_regex = re.compile(r'([a-zA-Z0-9_]+)\((.*?)\)')

    def get_predicate(self, name, arity):
        """Gets or creates a Z3 function for a predicate, keyed by (name, arity)."""
        key = (name, arity)
        if key not in self.predicates:
            sorts = [self.Entity] * arity + [BoolSort()]
            self.predicates[key] = Function(name, *sorts)
        return self.predicates[key]

    def get_constant(self, name):
        """Gets or creates a Z3 constant for an entity."""
        if name not in self.constants:
            self.constants[name] = Const(name, self.Entity)
        return self.constants[name]

    def _find_main_operator(self, expr_str):
        """Finds the main logical operator in an expression, respecting parentheses."""
        expr_str = expr_str.strip()
        paren_level = 0
        operators = ['->', '|', '&']  # In increasing order of precedence

        for op in operators:
            i = len(expr_str) - 1
            while i >= 0:
                char = expr_str[i]
                if char == ')':
                    paren_level += 1
                elif char == '(':
                    paren_level -= 1

                op_len = len(op)
                if paren_level == 0 and i >= op_len - 1 and expr_str[i - op_len + 1:i + 1] == op:
                    return i - op_len + 1, op
                i -= 1
        return -1, None

    def _parse_expression(self, expr_str, var_map):
        """Recursively parses a logical expression string into a Z3 expression."""
        expr_str = expr_str.strip()

        if expr_str.lower() == 'true': return BoolVal(True)
        if expr_str.lower() == 'false': return BoolVal(False)

        if expr_str.startswith('(') and expr_str.endswith(')'):
            paren_level = 0
            is_wrapping = True
            for i, char in enumerate(expr_str[:-1]):
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                if paren_level == 0 and i < len(expr_str) - 2:
                    is_wrapping = False
                    break
            if is_wrapping:
                return self._parse_expression(expr_str[1:-1], var_map)

        if expr_str.startswith('-'):
            return Not(self._parse_expression(expr_str[1:], var_map))

        op_idx, op_str = self._find_main_operator(expr_str)
        if op_str and op_idx >= 0:
            left_str = expr_str[:op_idx].strip()
            right_str = expr_str[op_idx + len(op_str):].strip()
            left = self._parse_expression(left_str, var_map)
            right = self._parse_expression(right_str, var_map)
            if op_str == '->': return Implies(left, right)
            if op_str == '|': return Or(left, right)
            if op_str == '&': return And(left, right)

        match = self.term_regex.match(expr_str)
        if match:
            pred_name, args_str = match.groups()
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]

            z3_args = []
            for arg in args:
                if arg in var_map:
                    z3_args.append(var_map[arg])
                else:
                    z3_args.append(self.get_constant(arg))

            predicate = self.get_predicate(pred_name, len(z3_args))
            return predicate(*z3_args)

        if expr_str in var_map:
            return var_map[expr_str]

        if expr_str:
            const = self.get_constant(expr_str)
            return const == const

        raise ValueError(f"Could not parse expression: '{expr_str}'")

    def parse_line(self, line):
        """Parses a single line, handling negations and multiple quantifiers."""
        line = line.strip().rstrip('.')

        is_negated = False
        # Handle potential negation at the start of any formula type
        if line.startswith('-'):
            # This is a bit simplistic; assumes negation isn't part of a name
            # A more robust way might check for '- ' or '-('
            is_negated = True
            line = line[1:].strip()

        quantifiers = []
        var_map = {}
        q_pattern = re.compile(r'^\s*(all|exists)\s+([a-z][a-zA-Z0-9_]*)\b(.*)')

        match = q_pattern.match(line)
        while match:
            q_type, var_name, rest_of_line = match.groups()
            z3_var = Const(var_name, self.Entity)
            var_map[var_name] = z3_var
            quantifiers.append({'type': q_type, 'var': z3_var})
            line = rest_of_line.strip()
            match = q_pattern.match(line)

        body_expr_str = line

        # If no quantifiers were found, we are parsing a simple expression
        if not quantifiers:
            # Re-add the negation for _parse_expression to handle it
            if is_negated:
                return self._parse_expression('-' + body_expr_str, {})
            else:
                return self._parse_expression(body_expr_str, {})

        # If we found quantifiers
        else:
            parsed_body = self._parse_expression(body_expr_str, var_map)
            result = parsed_body
            for q in reversed(quantifiers):
                if q['type'] == 'all':
                    result = ForAll([q['var']], result)
                else:  # 'exists'
                    result = Exists([q['var']], result)

            # Apply negation to the entire quantified expression
            if is_negated:
                return Not(result)
            else:
                return result

    def pre_scan(self, text):
        """Scans the text to declare all predicates and constants beforehand."""
        # Find all Predicate(...) patterns to pre-declare with correct arity
        for pred_name, args_str in self.term_regex.findall(text):
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            self.get_predicate(pred_name, len(args))
            for arg in args:
                # Pre-declare constants (lowercase names not in var_map)
                if arg and arg[0].islower():
                    self.get_constant(arg)

        # Find standalone constants (those not in predicates)
        all_terms = re.findall(r'\b[a-z][a-zA-Z0-9_]+\b', text)
        quantified_vars = re.findall(r'(?:all|exists)\s+([a-z][a-zA-Z0-9_]*)', text)
        for term in set(all_terms):  # Use set to avoid redundant calls
            if term not in quantified_vars:
                self.get_constant(term)

def verify_logic_problem(logic_text):
    """
    Solves a single logic problem from the JSON object.
    Returns 'A' (True), 'B' (False), or 'C' (Uncertain).
    """
    #logic_text = problem['raw_logic_programs'][0]

    try:
        assumptions_text = re.search(r'formulas\(assumptions\)\.(.*?)\nend_of_list\.', logic_text, re.DOTALL).group(1)
        goals_text = re.search(r'formulas\(goals\)\.(.*?)\nend_of_list\.', logic_text, re.DOTALL).group(1)
    except AttributeError:
        # print(f"ERROR: Could not parse assumptions/goals for ID {problem['id']}")
        return None

    parser = LogicParser()
    parser.pre_scan(logic_text)
    solver = Solver()

    assumption_lines = [line.strip() for line in assumptions_text.strip().split('\n') if line.strip()]
    for line in assumption_lines:
        try:
            parsed_expr = parser.parse_line(line)
            solver.add(parsed_expr)
        except Exception as e:
            # print(f"Warning: Failed to parse assumption line in {problem['id']}: '{line}' -> {e}")
            continue

    goal_line = goals_text.strip()
    try:
        goal = parser.parse_line(goal_line)
    except Exception as e:
        # print(f"ERROR: Failed to parse goal in {problem['id']}: '{goal_line}' -> {e}")
        return None

    # 1. Check if the goal must be TRUE
    solver.push()
    solver.add(Not(goal))
    result = solver.check()
    solver.pop()
    if result == unsat:
        return 'TRUE'

    # 2. Check if the goal must be FALSE
    solver.push()
    solver.add(goal)
    result = solver.check()
    solver.pop()
    if result == unsat:
        return 'FALSE'

    # 3. Otherwise, the goal is UNCERTAIN
    return 'UNCERTAIN'
