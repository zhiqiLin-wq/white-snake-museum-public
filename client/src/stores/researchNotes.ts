import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface ResearchNote {
  id: string
  locationName: string
  noteText: string
  createdAt: number
  updatedAt: number
}

const STORAGE_KEY = 'white_snake_research_notes'

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

function loadNotes(): ResearchNote[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) return parsed
    }
  } catch {
    // ignore
  }
  return []
}

function saveNotes(notes: ResearchNote[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes))
  } catch {
    // ignore
  }
}

export const useResearchNotesStore = defineStore('researchNotes', () => {
  const notes = ref<ResearchNote[]>(loadNotes())

  const locationNotes = computed(() => {
    const map = new Map<string, ResearchNote[]>()
    for (const note of notes.value) {
      const existing = map.get(note.locationName) || []
      existing.push(note)
      map.set(note.locationName, existing)
    }
    return map
  })

  function addNote(locationName: string, noteText?: string) {
    const existing = notes.value.find(n => n.locationName === locationName && !n.noteText)
    if (existing) return // dedup empty notes for same location

    const now = Date.now()
    notes.value.push({
      id: genId(),
      locationName,
      noteText: noteText || '',
      createdAt: now,
      updatedAt: now,
    })
    saveNotes(notes.value)
  }

  function removeNote(id: string) {
    notes.value = notes.value.filter(n => n.id !== id)
    saveNotes(notes.value)
  }

  function updateNote(id: string, noteText: string) {
    const note = notes.value.find(n => n.id === id)
    if (!note) return
    note.noteText = noteText
    note.updatedAt = Date.now()
    saveNotes(notes.value)
  }

  function hasNoteForLocation(locationName: string): boolean {
    return notes.value.some(n => n.locationName === locationName)
  }

  function clearAllNotes() {
    notes.value = []
    saveNotes(notes.value)
  }

  function exportNotesText(): string {
    let text = '研究笔记\n' + '='.repeat(30) + '\n\n'
    const byLocation = locationNotes.value
    for (const [loc, locNotes] of byLocation.entries()) {
      text += `## ${loc}\n`
      for (const note of locNotes) {
        text += `- [${new Date(note.createdAt).toLocaleString('zh-CN')}]`
        if (note.noteText) text += ` ${note.noteText}`
        text += '\n'
      }
      text += '\n'
    }
    text += '--- End ---'
    return text
  }

  return {
    notes,
    locationNotes,
    addNote,
    removeNote,
    updateNote,
    hasNoteForLocation,
    clearAllNotes,
    exportNotesText,
  }
})
