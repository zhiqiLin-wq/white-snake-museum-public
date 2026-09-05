export function escapeHtml(s: string | undefined | null): string {
  return s?.replace(/[&<>]/g, m =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[m as '&' | '<' | '>'])
  ) || ''
}

export function highlightText(text: string, queryWords: string[]): string {
  if (!queryWords || queryWords.length === 0) return escapeHtml(text)
  let escaped = escapeHtml(text)
  queryWords.forEach(qw => {
    const escapedQw = escapeHtml(qw)
    if (escapedQw.length === 0) return
    const regex = new RegExp(`(${escapedQw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    escaped = escaped.replace(regex, '<mark class="search-highlight">$1</mark>')
  })
  return escaped
}
