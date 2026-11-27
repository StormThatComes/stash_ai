"""Screen-sized role labels for the performer graph canvas."""
from .inspection_graph_layout import node_role


ROLE_COLORS = {
    'view': '#047857',
    'anchor': '#1d4ed8',
    'auxiliary': '#984b06',
    'attachment': '#6d28d9',
    'reference': '#475569',
    'unknown': '#64748b',
}


def role_labels(node):
    if node['type'] == 'group':
        counts = node['role_counts']
        return [(role, f'{role.title()} ×{counts[role]}') for role in ROLE_COLORS if role in counts]
    role = node_role(node)
    return [(role, role.title())] if role else []


class NodeCaption:
    """Lay out a caption once, then move all its canvas items together."""
    def __init__(self, canvas, shape, node, font, *, has_thumbnail):
        self.canvas = canvas
        self.tag = f'caption-{shape}'
        self.items = []
        self.x = self.y = 0
        badges = role_labels(node)
        height = font.metrics('linespace') + 2
        top = 0
        for start in range(0, len(badges), 2):
            row = badges[start:start + 2]
            widths = [font.measure(label) + 10 for role, label in row]
            left = -(sum(widths) + 3 * (len(row) - 1)) / 2
            for (role, label), width in zip(row, widths):
                self.items.append(canvas.create_rectangle(
                    left, top, left + width, top + height,
                    fill=ROLE_COLORS[role], outline='', tags=self.tag))
                self.items.append(canvas.create_text(
                    left + width / 2, top + height / 2, text=label,
                    font=font, fill='white', tags=self.tag))
                left += width + 3
            top += height + 2

        detail = ''
        if node['type'] == 'group':
            detail = node['label'] + (' · sample' if has_thumbnail else '')
        elif node['type'] == 'observation':
            detail = node['kind'].title()
            if node.get('scene') is not None and node.get('cue') is not None:
                detail += f" {node['scene']}:{node['cue']}"
        if detail:
            self.items.append(canvas.create_text(
                0, top, text=detail, anchor='n', width=155, justify='center',
                font=('TkDefaultFont', 9), fill='#0f172a', tags=self.tag))

    def place(self, x, y):
        self.canvas.move(self.tag, x - self.x, y - self.y)
        self.x, self.y = x, y
