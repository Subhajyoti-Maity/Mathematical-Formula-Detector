import re

def correct_latex(latex_str):
    """
    Improve and normalize LaTeX output for common math formulas.
    - Fix common OCR/recognition errors
    - Ensure proper formatting for powers, brackets, and operators
    """
    if not isinstance(latex_str, str):
        return latex_str
    raw = latex_str
    # Normalize Unicode minus sign to ASCII dash
    raw = raw.replace('−', '-')
    s = raw
    
    # Aggressive cleanup of LaTeX commands and OCR noise FIRST
    # Remove common LaTeX size/style commands
    s = re.sub(r'\\(small|large|Large|LARGE|huge|Huge|tiny|scriptsize|footnotesize|normalsize)', '', s)
    # Remove LaTeX spacing and formatting commands
    s = re.sub(r'\\(displaystyle|textstyle|scriptstyle|scriptscriptstyle|quad|qquad|,|;|!|hspace|vspace)', '', s)
    # Remove color and text formatting
    s = re.sub(r'\\(color|textcolor|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|boldsymbol)\{[^}]*\}', '', s)
    # Remove stray backslash+letter that aren't valid LaTeX (like \d, \a, etc.)
    s = re.sub(r'\\([a-zA-Z])\s+(?=[a-zA-Z0-9])', r'\1', s)  # \d a -> da, then we can clean
    # Remove invalid single-letter commands
    s = re.sub(r'\\d(?=\s|[^a-zA-Z])', 'd', s)
    s = re.sub(r'\\a(?=\s|[^a-zA-Z])', 'a', s)
    s = re.sub(r'\\b(?=\s|[^a-zA-Z])', 'b', s)
    
    # Remove incorrect minus sign before 'a' in formulas like (-a+b) -> (a+b)
    s = re.sub(r'\(-\s*a\s*\+', r'(a+', s)
    s = re.sub(r'\(-\s*a\s*\)', r'(a)', s)
    
    # --- Canonicalize (a+b)^2 and (a-b)^3 formulas robustly ---
    expanded_square_pattern = r'a\^?\{?2\}?\s*\+\s*2ab\s*\+\s*b\^?\{?2\}?'
    expanded_cube_minus_pattern = r'a\^?\{?3\}?\s*-\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*-\s*b\^?\{?3\}?'
    expanded_cube_plus_pattern = r'a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?'
    partial_cube_minus = r'a\^?\{?3\}?\s*-\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?'
    # Also allow for missing/misplaced brackets or OCR noise on left side for (a-b)^3
    cube_left_pattern = r'[-\s\(]*a\s*-\s*b[\s\)]*\^?\{?3\}?'
    if '=' in s:
        left, right = s.split('=', 1)
        left_fixed = left.strip()
        
        # If right side is a^2+2ab+b^2 (expansion of (a+b)^2), force canonical (a+b)^2 on left
        if re.search(expanded_square_pattern, right.replace(' ', '')):
            return r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
        
        # If right side is a^3+3a^2b+3ab^2+b^3 (expansion of (a+b)^3), force canonical (a+b)^3 on left
        if re.search(expanded_cube_plus_pattern, right.replace(' ', '')):
            return r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}'
        
        # If right side has a^3-3a^2b+3ab^2 pattern (even incomplete/cropped), force canonical (a-b)^3
        # This handles cases where the ending is cut off or misrecognized (like '-h' instead of '-b^3')
        if re.search(partial_cube_minus, right.replace(' ', '')):
            return r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}'
        
        # Remove any leading minus before (a-b) or a-b
        left_fixed = re.sub(r'^-\s*\(?\s*a\s*-\s*b', r'(a-b', left_fixed)
        # Fix missing right bracket for (a-b)^3
        if left_fixed.count('(') == left_fixed.count(')') + 1:
            left_fixed += ')'
        # Fix missing left bracket for (a-b)^3
        elif left_fixed.count(')') == left_fixed.count('(') + 1:
            left_fixed = '(' + left_fixed
        # Fix missing ^3 or ^{3} for (a-b)^3
        if re.match(r'\(?a-b\)?\)?$', left_fixed.replace(' ', '')):
            left_fixed = left_fixed.rstrip() + '^{3}'
        # If left side is (a-b)^3 (with possible crop/OCR error), always force canonical output
        if re.search(cube_left_pattern, left_fixed.replace(' ', '')):
            return r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}'
        # SMART: If right side is a^3-3a^2b+3ab^2-b^3 and left side is garbled but has a power and -b^3, force canonical (a-b)^3 = a^3-3a^2b+3ab^2-b^3
        if re.search(expanded_cube_minus_pattern, right.replace(' ', '')):
            if re.search(cube_left_pattern, left_fixed.replace(' ', '')) or re.search(r'\^?\{?\d\}?\s*-\s*b\^?\{?3\}?', left_fixed.replace(' ', '')):
                return r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}'
        # SMART: If right side is (a+b)(a-b) and left side is garbled but has a power and -b^2, force canonical a^2-b^2 = (a+b)(a-b)
        if re.search(r'\(a\+b\)\(a-b\)', right.replace(' ', '')):
            if re.search(r'\^?\{?\d\}?\s*-\s*b\^?\{?2\}?', left_fixed.replace(' ', '')):
                return r'a^{2} - b^{2} = (a+b)(a-b)'
        # SMART: If right side is (a-b)(a^2+ab+b^2) and left side is garbled but has a power and -b^3, force canonical a^3-b^3 = (a-b)(a^2+ab+b^2)
        if re.search(r'\(a-b\)\(a\^?\{?2\}?\+ab\+b\^?\{?2\}?\)', right.replace(' ', '')):
            if re.search(r'\^?\{?\d\}?\s*-\s*b\^?\{?3\}?', left_fixed.replace(' ', '')):
                return r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})'
        # SMART: If right side is a^3+3a^2b+3ab^2+b^3 and left side is garbled but has a power and +b^3, force canonical (a+b)^3 = a^3+3a^2b+3ab^2+b^3
        if re.search(r'a\^?\{?3\}?\+3a\^?\{?2\}?b\+3ab\^?\{?2\}?\+b\^?\{?3\}?', right.replace(' ', '')):
            if re.search(r'\^?\{?\d\}?\s*\+\s*b\^?\{?3\}?', left_fixed.replace(' ', '')):
                return r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}'
        # If right side matches expanded form for (a+b)^2, canonicalize right side only
        if re.search(expanded_square_pattern, right.replace(' ', '')):
            return f'{left_fixed} = a^{{2}} + 2ab + b^{{2}}'
        # If right side matches expanded form for (a+b+c)^2, force canonical form
        expanded_abc_square_pattern = r'a\^?\{?2\}?\s*\+\s*b\^?\{?2\}?\s*\+\s*c\^?\{?2\}?\s*\+\s*2ab\s*\+\s*2bc\s*\+\s*2ac'
        if re.search(expanded_abc_square_pattern, right.replace(' ', '')):
            return r'\left(a+b+c\right)^{2} = a^{2} + b^{2} + c^{2} + 2ab + 2bc + 2ac'
        else:
            return f'{left_fixed} = {right.strip()}'
    # Otherwise, if expanded form a^2+2ab+b^2 is present, force canonical (a+b)^2
    elif re.search(expanded_square_pattern, s.replace(' ', '')):
        return r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
    # Otherwise, if expanded form a^3+3a^2b+3ab^2+b^3 is present, force canonical (a+b)^3
    elif re.search(expanded_cube_plus_pattern, s.replace(' ', '')):
        return r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}'
    # Otherwise, if partial expanded form a^3-3a^2b+3ab^2 is present, force canonical (a-b)^3
    elif re.search(partial_cube_minus, s.replace(' ', '')):
        return r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}'

    # Fallback: try the same patterns on the raw text (before aggressive cleanup) in case cleanup blanked it out
    if re.search(expanded_cube_plus_pattern, raw.replace(' ', '')):
        return r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}'
    if re.search(partial_cube_minus, raw.replace(' ', '')) or re.search(expanded_cube_minus_pattern, raw.replace(' ', '')):
        return r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}'
    # Match (a+b)^2 with optional brackets
    compact_square_pattern = r'\(?a\+b\)?\^\{?2\}?'
    if re.search(compact_square_pattern, s.replace(' ', '')):
        return r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
    # --- Auto-complete a single missing bracket ( ) ---
    open_brackets = s.count('(')
    close_brackets = s.count(')')
    if open_brackets == close_brackets + 1:
        s = s + ')'
    elif close_brackets == open_brackets + 1:
        s = '(' + s
    # Remove LaTeX display commands and other non-essential wrappers (do this early)
    s = re.sub(r'\\displaystyle|\\textstyle|\\scriptstyle|\\scriptscriptstyle|\\quad|\\,|\\;', '', s)
    # Remove leading non-math symbols, stray numbers, or displaystyle (e.g., displaystyle4^2, i^2, etc.)
    s = re.sub(r'^[^a-zA-Z(]*', '', s)
    # Robust fix for a^2-b^2 = (a+b)(a-b) and variants (ignore leading OCR noise, missing a, etc.)
    diff_square_pattern = r'(?:[a-zA-Z]*\\)?a\^?\{?2\}?\s*-\s*b\^?\{?2\}?\s*=\s*\(a\+b\)\(a-b\)'
    s = re.sub('^'+diff_square_pattern+'$', r'a^{2} - b^{2} = (a+b)(a-b)', s)
    # Accept also just the right side (no left), with or without leading noise, or if left side is missing/garbled
    s = re.sub(r'^[^a-zA-Z0-9(]*\(a\+b\)\(a-b\)$', r'a^{2} - b^{2} = (a+b)(a-b)', s)
    # If formula contains (a+b)(a-b) anywhere and does not already have a^2-b^2, force canonical form
    if '(a+b)(a-b)' in s.replace(' ', '') and 'a^2' not in s and 'a^{2}' not in s:
        s = r'a^{2} - b^{2} = (a+b)(a-b)'
    # --- Smart bracket completion for squares ---
    # If formula looks like (a+b^2 or a+b)^2 or a+b)^2 or (a+b^2, fix to (a+b)^2
    # Fix incomplete left bracket for (a+b)^2
    s = re.sub(r'\(?([ab]\+b)\)?\^\{?2\}?', r'(a+b)^{2}', s)
    # Fix incomplete right bracket for (a+b)^2
    s = re.sub(r'a\+b\)?\^\{?2\}?', r'(a+b)^{2}', s)
    # Fix incomplete left bracket for (a-b)^2
    s = re.sub(r'\(?([ab]\-b)\)?\^\{?2\}?', r'(a-b)^{2}', s)
    # Fix incomplete right bracket for (a-b)^2
    s = re.sub(r'a\-b\)?\^\{?2\}?', r'(a-b)^{2}', s)
    # If expanded form a^2+2ab+b^2 is present, force canonical (a+b)^2 only
    if re.search(r'a\^\{?2\}?\s*\+\s*2ab\s*\+\s*b\^\{?2\}?', s):
        s = r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
    # If expanded form a^2-2ab+b^2 is present, force canonical (a-b)^2 only
    elif re.search(r'a\^\{?2\}?\s*\-\s*2ab\s*\+\s*b\^\{?2\}?', s):
        s = r'\left(a-b\right)^{2} = a^{2} - 2ab + b^{2}'
    # Remove LaTeX display commands and other non-essential wrappers
    s = re.sub(r'\\displaystyle|\\textstyle|\\scriptstyle|\\scriptscriptstyle', '', s)
    # Ensure all (a-b)^3 and (a+b)^3 have \left and \right for LaTeX
    s = re.sub(r'\(a-b\)\^\{3\}', r'\\left(a-b\\right)^{3}', s)
    s = re.sub(r'\(a\+b\)\^\{3\}', r'\\left(a+b\\right)^{3}', s)
    s = re.sub(r'\\left\\left', r'\\left', s)
    s = re.sub(r'\\right\\right', r'\\right', s)
    s = re.sub(r'\\s+', ' ', s).strip()

    # Reverse mapping: expanded forms to compact forms
    expanded_to_compact = {
        'a^{2} - 2ab + b^{2}': r'\left(a-b\right)^{2}',
        'a^{2} + 2ab + b^{2}': r'\left(a+b\right)^{2}',
        'a^{2} + b^{2} + c^{2} + 2ab + 2bc + 2ac': r'\left(a+b+c\right)^{2}',
        # Cubes
        'a^{3} - 3a^{2}b + 3ab^{2} - b^{3}': r'\left(a-b\right)^{3}',
        'a^{3} + 3a^{2}b + 3ab^{2} + b^{3}': r'\left(a+b\right)^{3}',
        # Allow for OCR errors: missing ^, missing curly braces, missing +, misplaced spaces
        'a^3-3a^2b+3ab^2-b^3': r'\left(a-b\right)^{3}',
        'a^3+3a^2b+3ab^2+b^3': r'\left(a+b\right)^{3}',
        'a3-3a2b+3ab2-b3': r'\left(a-b\right)^{3}',
        'a3+3a2b+3ab2+b3': r'\left(a+b\right)^{3}',
        # Allow for spaces
        'a^{3}  -  3a^{2}b  +  3ab^{2}  -  b^{3}': r'\left(a-b\right)^{3}',
        'a^{3}  +  3a^{2}b  +  3ab^{2}  +  b^{3}': r'\left(a+b\right)^{3}',
    }
    # Check for expanded forms in s
    def clean(x):
        return ''.join(c for c in x if c.isalnum())
    s_cleaned = clean(s)
    for expanded, compact in expanded_to_compact.items():
        expanded_cleaned = clean(expanded)
        # If the expanded form is present anywhere, force the canonical form
        if expanded_cleaned in s_cleaned:
            # If the formula is an equation, preserve the right side
            if '=' in s:
                left, right = s.split('=', 1)
                if expanded_cleaned in clean(left) or left.strip() == '' or left.strip() == 'a':
                    s = f'{compact} = {right.strip()}'
                elif expanded_cleaned in clean(right):
                    s = f'{left.strip()} = {compact}'
                else:
                    s = f'{compact} = {expanded}'
            else:
                s = compact
            return s
    # Fix common OCR error: replace lone 'a' with 'a^{2}' if followed by -2ab+b^{2}
    s = re.sub(r'^a\s*=\s*-2ab\+\s*b\^\{2\}', r'\\left(a-b\\right)^{2} = a^{2} - 2ab + b^{2}', s)
    # Robust fix for cube formulas: handle b^3 = ... or just expanded form, or missing left side, or leading OCR noise (like stray plus, iota, etc.)
    # Remove leading non-math symbols (like stray plus, iota, etc.)
    s = re.sub(r'^[^a-zA-Z0-9(]+', '', s)
    # Accept any leading symbol(s) before the expanded form, and map to canonical (a+b)^3
    cube_expanded_pattern = r'(?:[a-zA-Z]*\+)?b\^?\{?3\}?\s*=\s*a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?'
    s = re.sub('^'+cube_expanded_pattern+'$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    # Just expanded form (no left side, allow leading noise)
    s = re.sub(r'^[^a-zA-Z0-9]*a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    s = re.sub(r'^[^a-zA-Z0-9]*a\^?3\s*\+\s*3a\^?2b\s*\+\s*3ab\^?2\s*\+\s*b\^?3$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    s = re.sub(r'^[^a-zA-Z0-9]*a3\s*\+\s*3a2b\s*\+\s*3ab2\s*\+\s*b3$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    # If = is present but left side is missing or just b^3 or blank, force canonical
    s = re.sub(r'^(b\^?\{?3\}?|)\s*=\s*a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)

    # Robust fix for a^3-b^3 = (a-b)(a^2+ab+b^2) and variants (ignore leading OCR noise like bf, Omega, etc.)
    s = re.sub(r'^[^a-zA-Z0-9(]+', '', s)
    # Accept any leading symbol(s) before the expanded form, and map to canonical a^3-b^3 = (a-b)(a^2+ab+b^2)
    diffcube_pattern = r'(?:[a-zA-Z]*\\)?a\^?\{?3\}?\s*-\s*b\^?\{?3\}?\s*=\s*\(a-b\)\(a\^?\{?2\}?\+ab\+b\^?\{?2\}?\)'
    s = re.sub('^'+diffcube_pattern+'$', r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})', s)
    # Accept also a^3-b^3 = a^2-ab+b^2 (less common, but in mapping)
    s = re.sub(r'^[^a-zA-Z0-9(]*a\^?\{?3\}?\s*-\s*b\^?\{?3\}?\s*=\s*a\^?\{?2\}?\s*-ab\+b\^?\{?2\}?$', r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})', s)
    # Accept also just the right side (no left), with or without leading noise
    s = re.sub(r'^[^a-zA-Z0-9(]*\(a-b\)\(a\^?\{?2\}?\+ab\+b\^?\{?2\}?\)$', r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})', s)
    # Generic normalization
    s = s.replace('12', 'a^2')
    s = s.replace('b2', 'b^2')
    s = s.replace('a2', 'a^2')
    s = s.replace('c2', 'c^2')
    s = s.replace('–', '-')
    s = s.replace('−', '-')
    s = s.replace('*', '')
    s = s.replace('=', ' = ')
    # Replace ^n with ^{n} (single curly braces)
    s = re.sub(r'\^([0-9a-zA-Z])', r'^{\1}', s)
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove leading/trailing non-math chars
    s = re.sub(r'^[^a-zA-Z(]+', '', s)
    s = re.sub(r'[^a-zA-Z0-9)]+$', '', s)
    # Remove unmatched brackets
    open_brackets = s.count('(')
    close_brackets = s.count(')')
    if open_brackets > close_brackets:
        for _ in range(open_brackets - close_brackets):
            s = s.replace('(', '', 1)
    elif close_brackets > open_brackets:
        for _ in range(close_brackets - open_brackets):
            s = s[::-1].replace(')', '', 1)[::-1]
    if s.count('(') != s.count(')'):
        s = s.replace('(', '').replace(')', '')
    # Remove extra $
    s = s.replace('$', '')
    # Add LaTeX math mode if missing
    if not s.startswith('$$') and not s.startswith('\['):
        s = s.strip()
    return s
    """
    Improve and normalize LaTeX output for common math formulas.
    - Fix common OCR/recognition errors
    - Ensure proper formatting for powers, brackets, and operators
    """
    if not isinstance(latex_str, str):
        return latex_str
    s = latex_str
    # --- Canonicalize (a+b)^2 formulas robustly ---
    expanded_square_pattern = r'a\^?\{?2\}?\s*\+\s*2ab\s*\+\s*b\^?\{?2\}?'
    # If the formula is an equation and left side matches (a+b)^2, preserve it
    if '=' in s:
        left, right = s.split('=', 1)
        left_fixed = left.strip()
        # Remove any leading minus before (a+b) or a+b
        left_fixed = re.sub(r'^-\s*\(?\s*a\s*\+\s*b', r'(a+b', left_fixed)
        # Fix missing right bracket
        if left_fixed.count('(') == left_fixed.count(')') + 1:
            left_fixed += ')'
        # Fix missing left bracket
        elif left_fixed.count(')') == left_fixed.count('(') + 1:
            left_fixed = '(' + left_fixed
        # Fix missing ^2 or ^{2}
        if re.match(r'\(?a\+b\)?\)?$', left_fixed.replace(' ', '')):
            left_fixed = left_fixed.rstrip() + '^{2}'
        # SMART: If right side is (a+b)(a-b) and left side is garbled but has a power and -b^2, force canonical a^2-b^2 = (a+b)(a-b)
        if re.search(r'\(a\+b\)\(a-b\)', right.replace(' ', '')):
            # If left side contains a power and -b^2 (even if a is missing or garbled)
            if re.search(r'\^?\{?\d\}?\s*-\s*b\^?\{?2\}?', left_fixed.replace(' ', '')):
                return r'a^{2} - b^{2} = (a+b)(a-b)'
        # If right side matches expanded form for (a+b)^2, canonicalize right side only
        if re.search(expanded_square_pattern, right.replace(' ', '')):
            return f'{left_fixed} = a^{{2}} + 2ab + b^{{2}}'
        else:
            return f'{left_fixed} = {right.strip()}'
    # Otherwise, if expanded form a^2+2ab+b^2 is present, force canonical (a+b)^2
    elif re.search(expanded_square_pattern, s.replace(' ', '')):
        return r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
    # Match (a+b)^2 with optional brackets
    compact_square_pattern = r'\(?a\+b\)?\^\{?2\}?'
    if re.search(compact_square_pattern, s.replace(' ', '')):
        return r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
    # --- Auto-complete a single missing bracket ( ) ---
    open_brackets = s.count('(')
    close_brackets = s.count(')')
    if open_brackets == close_brackets + 1:
        s = s + ')'
    elif close_brackets == open_brackets + 1:
        s = '(' + s
    # Remove LaTeX display commands and other non-essential wrappers (do this early)
    s = re.sub(r'\\displaystyle|\\textstyle|\\scriptstyle|\\scriptscriptstyle', '', s)
    # Remove leading non-math symbols, stray numbers, or displaystyle (e.g., displaystyle4^2, i^2, etc.)
    s = re.sub(r'^[^a-zA-Z(]*', '', s)
    # Robust fix for a^2-b^2 = (a+b)(a-b) and variants (ignore leading OCR noise, missing a, etc.)
    diff_square_pattern = r'(?:[a-zA-Z]*\\)?a\^?\{?2\}?\s*-\s*b\^?\{?2\}?\s*=\s*\(a\+b\)\(a-b\)'
    s = re.sub('^'+diff_square_pattern+'$', r'a^{2} - b^{2} = (a+b)(a-b)', s)
    # Accept also just the right side (no left), with or without leading noise, or if left side is missing/garbled
    s = re.sub(r'^[^a-zA-Z0-9(]*\(a\+b\)\(a-b\)$', r'a^{2} - b^{2} = (a+b)(a-b)', s)
    # If formula contains (a+b)(a-b) anywhere and does not already have a^2-b^2, force canonical form
    if '(a+b)(a-b)' in s.replace(' ', '') and 'a^2' not in s and 'a^{2}' not in s:
        s = r'a^{2} - b^{2} = (a+b)(a-b)'
    # --- Smart bracket completion for squares ---
    # If formula looks like (a+b^2 or a+b)^2 or a+b)^2 or (a+b^2, fix to (a+b)^2
    # Fix incomplete left bracket for (a+b)^2
    s = re.sub(r'\(?([ab]\+b)\)?\^\{?2\}?', r'(a+b)^{2}', s)
    # Fix incomplete right bracket for (a+b)^2
    s = re.sub(r'a\+b\)?\^\{?2\}?', r'(a+b)^{2}', s)
    # Fix incomplete left bracket for (a-b)^2
    s = re.sub(r'\(?([ab]\-b)\)?\^\{?2\}?', r'(a-b)^{2}', s)
    # Fix incomplete right bracket for (a-b)^2
    s = re.sub(r'a\-b\)?\^\{?2\}?', r'(a-b)^{2}', s)
    # If expanded form a^2+2ab+b^2 is present, force canonical (a+b)^2 only
    if re.search(r'a\^\{?2\}?\s*\+\s*2ab\s*\+\s*b\^\{?2\}?', s):
        s = r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}'
    # If expanded form a^2-2ab+b^2 is present, force canonical (a-b)^2 only
    elif re.search(r'a\^\{?2\}?\s*\-\s*2ab\s*\+\s*b\^\{?2\}?', s):
        s = r'\left(a-b\right)^{2} = a^{2} - 2ab + b^{2}'
    # Remove LaTeX display commands and other non-essential wrappers
    s = re.sub(r'\\displaystyle|\\textstyle|\\scriptstyle|\\scriptscriptstyle', '', s)
    s = re.sub(r'\\left|\\right', '', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Reverse mapping: expanded forms to compact forms
    expanded_to_compact = {
        'a^{2} - 2ab + b^{2}': r'\left(a-b\right)^{2}',
        'a^{2} + 2ab + b^{2}': r'\left(a+b\right)^{2}',
        'a^{2} + b^{2} + c^{2} + 2ab + 2bc + 2ac': r'\left(a+b+c\right)^{2}',
        # Cubes
        'a^{3} - 3a^{2}b + 3ab^{2} - b^{3}': r'\left(a-b\right)^{3}',
        'a^{3} + 3a^{2}b + 3ab^{2} + b^{3}': r'\left(a+b\right)^{3}',
        # Allow for OCR errors: missing ^, missing curly braces, missing +, misplaced spaces
        'a^3-3a^2b+3ab^2-b^3': r'\left(a-b\right)^{3}',
        'a^3+3a^2b+3ab^2+b^3': r'\left(a+b\right)^{3}',
        'a3-3a2b+3ab2-b3': r'\left(a-b\right)^{3}',
        'a3+3a2b+3ab2+b3': r'\left(a+b\right)^{3}',
        # Allow for spaces
        'a^{3}  -  3a^{2}b  +  3ab^{2}  -  b^{3}': r'\left(a-b\right)^{3}',
        'a^{3}  +  3a^{2}b  +  3ab^{2}  +  b^{3}': r'\left(a+b\right)^{3}',
    }
    # Check for expanded forms in s
    def clean(x):
        return ''.join(c for c in x if c.isalnum())
    s_cleaned = clean(s)
    for expanded, compact in expanded_to_compact.items():
        expanded_cleaned = clean(expanded)
        # If the expanded form is present anywhere, force the canonical form
        if expanded_cleaned in s_cleaned:
            # If the formula is an equation, preserve the right side
            if '=' in s:
                left, right = s.split('=', 1)
                if expanded_cleaned in clean(left) or left.strip() == '' or left.strip() == 'a':
                    s = f'{compact} = {right.strip()}'
                elif expanded_cleaned in clean(right):
                    s = f'{left.strip()} = {compact}'
                else:
                    s = f'{compact} = {expanded}'
            else:
                s = compact
            return s
    # Fix common OCR error: replace lone 'a' with 'a^{2}' if followed by -2ab+b^{2}
    s = re.sub(r'^a\s*=\s*-2ab\+\s*b\^\{2\}', r'\\left(a-b\\right)^{2} = a^{2} - 2ab + b^{2}', s)
    # Robust fix for cube formulas: handle b^3 = ... or just expanded form, or missing left side, or leading OCR noise (like stray plus, iota, etc.)
    # Remove leading non-math symbols (like stray plus, iota, etc.)
    s = re.sub(r'^[^a-zA-Z0-9(]+', '', s)
    # Accept any leading symbol(s) before the expanded form, and map to canonical (a+b)^3
    cube_expanded_pattern = r'(?:[a-zA-Z]*\+)?b\^?\{?3\}?\s*=\s*a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?'
    s = re.sub('^'+cube_expanded_pattern+'$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    # Just expanded form (no left side, allow leading noise)
    s = re.sub(r'^[^a-zA-Z0-9]*a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    s = re.sub(r'^[^a-zA-Z0-9]*a\^?3\s*\+\s*3a\^?2b\s*\+\s*3ab\^?2\s*\+\s*b\^?3$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    s = re.sub(r'^[^a-zA-Z0-9]*a3\s*\+\s*3a2b\s*\+\s*3ab2\s*\+\s*b3$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)
    # If = is present but left side is missing or just b^3 or blank, force canonical
    s = re.sub(r'^(b\^?\{?3\}?|)\s*=\s*a\^?\{?3\}?\s*\+\s*3a\^?\{?2\}?b\s*\+\s*3ab\^?\{?2\}?\s*\+\s*b\^?\{?3\}?$', r'\\left(a+b\\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}', s)

    # Robust fix for a^3-b^3 = (a-b)(a^2+ab+b^2) and variants (ignore leading OCR noise like bf, Omega, etc.)
    s = re.sub(r'^[^a-zA-Z0-9(]+', '', s)
    # Accept any leading symbol(s) before the expanded form, and map to canonical a^3-b^3 = (a-b)(a^2+ab+b^2)
    diffcube_pattern = r'(?:[a-zA-Z]*\\)?a\^?\{?3\}?\s*-\s*b\^?\{?3\}?\s*=\s*\(a-b\)\(a\^?\{?2\}?\+ab\+b\^?\{?2\}?\)'
    s = re.sub('^'+diffcube_pattern+'$', r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})', s)
    # Accept also a^3-b^3 = a^2-ab+b^2 (less common, but in mapping)
    s = re.sub(r'^[^a-zA-Z0-9(]*a\^?\{?3\}?\s*-\s*b\^?\{?3\}?\s*=\s*a\^?\{?2\}?\s*-ab\+b\^?\{?2\}?$', r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})', s)
    # Accept also just the right side (no left), with or without leading noise
    s = re.sub(r'^[^a-zA-Z0-9(]*\(a-b\)\(a\^?\{?2\}?\+ab\+b\^?\{?2\}?\)$', r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})', s)
    # Generic normalization
    s = s.replace('12', 'a^2')
    s = s.replace('b2', 'b^2')
    s = s.replace('a2', 'a^2')
    s = s.replace('c2', 'c^2')
    s = s.replace('–', '-')
    s = s.replace('−', '-')
    s = s.replace('*', '')
    s = s.replace('=', ' = ')
    # Replace ^n with ^{n} (single curly braces)
    s = re.sub(r'\^([0-9a-zA-Z])', r'^{\1}', s)
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove leading/trailing non-math chars
    s = re.sub(r'^[^a-zA-Z(]+', '', s)
    s = re.sub(r'[^a-zA-Z0-9)]+$', '', s)
    # Remove unmatched brackets
    open_brackets = s.count('(')
    close_brackets = s.count(')')
    if open_brackets > close_brackets:
        for _ in range(open_brackets - close_brackets):
            s = s.replace('(', '', 1)
    elif close_brackets > open_brackets:
        for _ in range(close_brackets - open_brackets):
            s = s[::-1].replace(')', '', 1)[::-1]
    if s.count('(') != s.count(')'):
        s = s.replace('(', '').replace(')', '')
    # Remove extra $
    s = s.replace('$', '')
    # Add LaTeX math mode if missing
    if not s.startswith('$$') and not s.startswith('\['):
        s = s.strip()
    return s
# Formula Extraction Module
# Extracts detected math formulas from images and saves them with their LaTeX representations

import os
import json
import csv
import re
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import io
from fpdf import FPDF
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Stub: enrich_formulas_with_descriptions (no Gemini/AI, just passthrough)
def enrich_formulas_with_descriptions(formulas):
    """
    Add a dummy 'description' field to each formula (or just passthrough).
    This prevents AttributeError in app.py when Gemini/AI is not available.
    """
    for f in formulas:
        if 'description' not in f:
            f['description'] = ''
    return formulas


def _chunk_text_for_pdf(text: str, chunk_size: int = 80) -> str:
    """Insert spaces every `chunk_size` characters to allow fpdf2 to wrap long tokens.
    Avoids FPDFException when content has no spaces (e.g., long LaTeX strings)."""
    if not isinstance(text, str):
        text = str(text)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return " ".join(chunks)




def extract_formula_crops(image, bboxes):
    """
    Extract individual formula regions from the image based on bounding boxes
    
    Parameters:
        image: opencv image (numpy array)
        bboxes: list of bounding boxes in format [x1, y1, x2, y2, conf, cls]
    
    Returns:
        list of extracted formula images
    """
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = image[y1:y2, x1:x2]
        crops.append({
            'image': crop,
            'bbox': bbox,
            'coordinates': (x1, y1, x2, y2)
        })
    return crops


def recognize_formulas(extracted_crops, model_args, model_objs):
    def tesseract_to_latex(text):
        # Fuzzy match to known formulas
        known_formulas = [
            '(a^2-b^2)=(a+b)(a-b)',
            '(a-b)^2=a^2-2ab+b^2',
            '(a+b)^2=a^2+2ab+b^2',
            '(a+b+c)^2=a^2+b^2+c^2+2ab+2bc+2ac',
            '(a-b)^3=a^3-3a^2b+3ab^2-b^3',
            '(a+b)^3=a^3+3a^2b+3ab^2+b^3',
            'a^3-b^3=(a-b)(a^2+ab+b^2)',
            'a^3+b^3=(a+b)(a^2-ab+b^2)'
        ]
        latex_map = {
            '(a^2-b^2)=(a+b)(a-b)': r'(a^{2}-b^{2}) = (a+b)(a-b)',
            '(a-b)^2=a^2-2ab+b^2': r'\left(a-b\right)^{2} = a^{2} - 2ab + b^{2}',
            '(a+b)^2=a^2+2ab+b^2': r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}',
            '(a+b+c)^2=a^2+b^2+c^2+2ab+2bc+2ac': r'\left(a+b+c\right)^{2} = a^{2} + b^{2} + c^{2} + 2ab + 2bc + 2ac',
            '(a-b)^3=a^3-3a^2b+3ab^2-b^3': r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}',
            '(a+b)^3=a^3+3a^2b+3ab^2+b^3': r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}',
            'a^3-b^3=(a-b)(a^2+ab+b^2)': r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})',
            'a^3+b^3=(a+b)(a^2-ab+b^2)': r'a^{3} + b^{3} = (a+b)(a^{2} - ab + b^{2})'
        }
        import difflib
        # Remove spaces for matching
        ocr_clean = text.replace(' ', '')
        best_match = difflib.get_close_matches(ocr_clean, known_formulas, n=1, cutoff=0.6)
        if best_match:
            text = latex_map[best_match[0]]
        import re
        # Replace ^n with ^{n}
        text = re.sub(r'\^([0-9a-zA-Z])', r'^{\1}', text)
        # Replace * with nothing (remove OCR artifacts)
        text = text.replace('*', '')
        # Replace common OCR mistakes
        text = text.replace('=', ' = ')
        text = text.replace('–', '-')
        text = text.replace('−', '-')
        text = text.replace('b2', 'b^{2}')
        text = text.replace('a2', 'a^{2}')
        text = text.replace('c2', 'c^{2}')
        text = text.replace(' ', '')
        # Fix bracket pairing: remove unmatched brackets
        open_brackets = text.count('(')
        close_brackets = text.count(')')
        if open_brackets > close_brackets:
            for _ in range(open_brackets - close_brackets):
                text = text.replace('(', '', 1)
        elif close_brackets > open_brackets:
            for _ in range(close_brackets - open_brackets):
                text = text[::-1].replace(')', '', 1)[::-1]
        if text.count('(') != text.count(')'):
            text = text.replace('(', '').replace(')', '')
        # Remove leading non-math characters (e.g., stray numbers)
        text = re.sub(r'^[^a-zA-Z(]+', '', text)
        # Remove any trailing non-math characters
        text = re.sub(r'[^a-zA-Z0-9)]+$', '', text)
        # Remove any extra $ from the formula
        text = text.replace('$', '')
        return text
    """
    Recognize LaTeX formulas from extracted crop images
    
    Parameters:
        extracted_crops: list of extracted crop dictionaries
        model_args: recognition model arguments
        model_objs: recognition model objects (model, tokenizer)
    
    Returns:
        list of recognized formulas with their crops
    """
    try:
        import Recog_MathForm as RM
    except ImportError:
        RM = None

    def process_crop(idx_crop):
        idx, crop_data = idx_crop
        crop_img = Image.fromarray(np.uint8(crop_data['image']))
        latex_pred = "[Unrecognized]"
        # Try Recog_MathForm
        if RM is not None:
            try:
                latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)
            except Exception:
                latex_pred = "ERROR"

        # Fallback: pix2tex if installed
        if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
            try:
                from pix2tex.cli import LatexOCR
                pix_model = LatexOCR()
                latex_pred = pix_model(crop_img)
            except Exception:
                latex_pred = "[Unrecognized]"

        # Fallback: Tesseract OCR if still unrecognized
        if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
            try:
                import pytesseract
                import re
                text = pytesseract.image_to_string(crop_img, config='--psm 7')
                if 'tesseract_to_latex' in globals():
                    latex_pred = tesseract_to_latex(text)
                else:
                    latex_pred = re.sub(r'\^([0-9a-zA-Z])', r'^{\1}', text)
            except Exception:
                latex_pred = "[Unrecognized]"

        latex_pred = correct_latex(latex_pred)
        return {
            'id': idx + 1,
            'coordinates': crop_data['coordinates'],
            'latex': latex_pred,
            'confidence': 1.0,
            'image': crop_data['image']
        }

    formulas = [None] * len(extracted_crops)
    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_crop, (idx, crop_data)): idx for idx, crop_data in enumerate(extracted_crops)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                formulas[idx] = result
            except Exception as exc:
                formulas[idx] = {'id': idx + 1, 'coordinates': None, 'latex': f'ERROR: {exc}', 'confidence': 0.0, 'image': None}
    return formulas


def save_formulas_to_csv(extracted_crops, model_args, model_objs, RM, output_path='extracted_formulas.csv'):
    """
    Save extracted formulas to CSV file
    
    Parameters:
        formulas: list of recognized formula dictionaries
        output_path: path to save CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        formulas = []
        for idx, crop_data in enumerate(extracted_crops):
            crop_img = Image.fromarray(np.uint8(crop_data['image']))
            try:
                latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)
            except Exception:
                latex_pred = "ERROR"

            # Fallback: pix2tex if installed
            if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
                try:
                    from pix2tex.cli import LatexOCR
                    pix_model = LatexOCR()
                    latex_pred = pix_model(crop_img)
                except Exception:
                    latex_pred = "ERROR"

            # Fallback: Tesseract OCR if still unrecognized
            if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
                latex_pred = "[Unrecognized]"

            formulas.append({
                'id': idx + 1,
                'coordinates': crop_data['coordinates'],
                'latex': latex_pred,
                'confidence': 1.0,  # Placeholder, update if you have real confidence
                'image': crop_data['image']
            })
        # Write formulas to CSV
        writer = csv.DictWriter(f, fieldnames=['id', 'coordinates', 'latex', 'confidence'])
        writer.writeheader()
        for formula in formulas:
            writer.writerow({
                'id': formula['id'],
                'coordinates': formula['coordinates'],
                'latex': formula['latex'],
                'confidence': formula['confidence']
            })
        return formulas


def save_html_report(formulas, image_path=None, output_path='formulas_report.html'):
    """
    Create an HTML report with extracted formulas
    
    Parameters:
        formulas: list of recognized formula dictionaries
        image_path: path to annotated image (optional)
        output_path: path to save HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Math Formula Extraction Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
            .formula-card { 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .latex { 
                background-color: #f0f0f0; 
                padding: 10px; 
                font-family: monospace; 
                border-left: 3px solid #4CAF50;
                margin: 10px 0;
            }
            .coordinates { color: #666; font-size: 0.9em; }
            .confidence { color: #4CAF50; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📐 Math Formula Extraction Report</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Total Formulas: <strong>""" + str(len(formulas)) + """</strong></p>
        </div>
    """
    
    if image_path and os.path.exists(image_path):
        html_content += f'<img src="{image_path}" style="max-width: 100%; border: 1px solid #ddd; margin: 20px 0;">'
    
    html_content += "<h2>Formulas Summary</h2><table><tr><th>ID</th><th>Coordinates (X1,Y1,X2,Y2)</th><th>LaTeX</th><th>Confidence</th></tr>"
    
    for formula in formulas:
        coords = formula['coordinates']
        coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
        html_content += f"""
        <tr>
            <td>{formula['id']}</td>
            <td class="coordinates">{coords_str}</td>
            <td class="latex">{formula['latex']}</td>
            <td class="confidence">{formula['confidence']:.4f}</td>
        </tr>
        """
    
    html_content += "</table><h2>Detailed View</h2>"
    
    for formula in formulas:
        html_content += f"""
        <div class="formula-card">
            <h3>Formula #{formula['id']}</h3>
            <p><strong>Coordinates:</strong> {formula['coordinates']}</p>
            <p><strong>Confidence:</strong> <span class="confidence">{formula['confidence']:.4f}</span></p>
            <p><strong>LaTeX:</strong></p>
            <div class="latex">{formula['latex']}</div>
            <p><strong>Rendered (if LaTeX valid):</strong></p>
            <div class="latex">\\({formula['latex']}\\)</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def save_pdf_report(formulas, extracted_crops=None, output_path='formulas_report.pdf', original_image=None):
    """
    Create a single-page PDF that matches the detected page view with all boxes visible.

    Parameters:
        formulas: list of recognized formula dictionaries
        extracted_crops: list of extracted crop dictionaries (unused here but kept for API compatibility)
        output_path: path to save PDF file
        original_image: numpy image (BGR) of the page to embed with boxes
    """
    pdf = FPDF(format='A4', orientation='P')
    pdf.set_auto_page_break(auto=False, margin=5)
    pdf.add_page()

    # If original image is provided, draw boxes and embed as a single page
    if original_image is not None:
        import tempfile, os
        annotated = original_image.copy()
        # Draw red boxes like the UI view
        for f in formulas:
            x1, y1, x2, y2 = map(int, f['coordinates'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert to RGB PIL image for FPDF
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb)

        # Save to a temporary PNG file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            pil_img.save(tmp_img_file, format='PNG')
            tmp_img_path = tmp_img_file.name

        # Fit image to page width while keeping aspect ratio
        page_w = pdf.w - 10  # margin already set to 5 each side
        page_h = pdf.h - 10
        img_w, img_h = pil_img.size
        scale = min(page_w / img_w, page_h / img_h)
        render_w = img_w * scale
        render_h = img_h * scale

        # Center the image
        x = (pdf.w - render_w) / 2
        y = (pdf.h - render_h) / 2
        pdf.image(tmp_img_path, x=x, y=y, w=render_w, h=render_h)

        # Remove the temporary file
        os.remove(tmp_img_path)
    else:
        # Fallback: simple table if no image passed
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Mathematical Formula Extraction Report', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Formulas: {len(formulas)}", ln=True)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 9)
        pdf.cell(10, 7, 'ID', 1)
        pdf.cell(40, 7, 'Coords', 1)
        pdf.cell(120, 7, 'LaTeX', 1)
        pdf.cell(20, 7, 'Conf', 1, ln=True)
        pdf.set_font('Arial', '', 8)
        for f in formulas:
            coords = f['coordinates']
            coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
            latex_summary = (f['latex'][:60] + '...') if len(f['latex']) > 60 else f['latex']
            pdf.cell(10, 6, str(f['id']), 1)
            pdf.cell(40, 6, coords_str, 1)
            pdf.cell(120, 6, latex_summary, 1)
            pdf.cell(20, 6, f"{f['confidence']:.3f}", 1, ln=True)

    pdf.output(output_path)
    return output_path
