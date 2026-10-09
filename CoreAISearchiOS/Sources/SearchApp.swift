// CoreAISearchiOS — search your own text by meaning, on an iPhone. Granite Embedding 97M runs on
// Apple's Core AI runtime (unmodified); the model comes from the Hugging Face Hub on the first launch.

import SwiftUI
import UniformTypeIdentifiers

@main
struct SearchApp: App {
    var body: some Scene {
        WindowGroup { SearchView() }
    }
}

// The root view sets ink as the foreground style, so a filled (borderedProminent) button sets its label white itself.
private let ink = Color(red: 0.12, green: 0.17, blue: 0.17)
private let paper = Color(red: 0.97, green: 0.96, blue: 0.93)

private func megabytes(_ bytes: Int64) -> String {
    (Double(bytes) / 1_000_000).formatted(.number.precision(.fractionLength(0)))
}

struct SearchView: View {
    @StateObject private var model = SearchModel()
    @State private var expanded: String?
    @State private var confirmClear = false
    @FocusState private var searchFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            switch model.phase {
            case .checking, .loading: loadingContent
            case .needsDownload, .downloading, .downloadFailed: downloadContent
            case .indexing: indexingContent
            case .ready: searchContent
            case .failed(let message): failedContent(message)
            }
        }
        .foregroundStyle(ink)
        .background(paper.ignoresSafeArea())
        .preferredColorScheme(.light)
        .task { model.start() }
        .sheet(isPresented: $model.isAddSheetPresented) { AddTextSheet(model: model) }
        .confirmationDialog("Remove the \(model.notes.count.formatted()) sentences you added?",
                            isPresented: $confirmClear, titleVisibility: .visible) {
            Button("Clear my notes", role: .destructive) { Task { await model.clearNotes() } }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Core AI Search").font(.system(.title2, design: .serif, weight: .bold))
                Text("Find a sentence by what it means").font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if model.phase == .ready {
                Button { model.presentAddSheet() } label: {
                    Label("Add", systemImage: "plus").font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 12).padding(.vertical, 7)
                        .background(.white, in: Capsule())
                }
                .accessibilityLabel("Add your text")
                Menu {
                    Button("Clear my notes", systemImage: "trash", role: .destructive) { confirmClear = true }
                        .disabled(model.notes.isEmpty)
                } label: {
                    Image(systemName: "ellipsis.circle").font(.title3)
                }
            }
        }
        .padding(.horizontal, 20).padding(.top, 16).padding(.bottom, 16)
    }

    // MARK: Before the search

    private var downloadContent: some View {
        VStack(alignment: .leading, spacing: 22) {
            Spacer(minLength: 12)
            Image(systemName: "arrow.down.circle").font(.system(size: 44, weight: .light))
            Text("Search your notes by meaning").font(.system(size: 31, weight: .medium, design: .serif))
            Text("Granite Embedding 97M runs on this iPhone with Core AI. The app downloads the model once from Hugging Face; your text stays on this iPhone.")
                .font(.callout).foregroundStyle(.secondary)
            switch model.phase {
            case .downloading:
                VStack(alignment: .leading, spacing: 10) {
                    ProgressView(value: Double(model.downloadedBytes), total: Double(ModelFiles.totalBytes)).tint(ink)
                    Text("Downloading \(megabytes(model.downloadedBytes)) of \(megabytes(ModelFiles.totalBytes)) MB")
                        .font(.footnote.monospacedDigit())
                }
                .accessibilityIdentifier("download-progress")
            case .downloadFailed(let message):
                VStack(alignment: .leading, spacing: 10) {
                    Label("Could not download the model", systemImage: "exclamationmark.triangle").font(.headline)
                    Text(message).font(.callout).fixedSize(horizontal: false, vertical: true)
                    Text("Files that finished are kept; Retry downloads the rest.").font(.footnote).foregroundStyle(.secondary)
                    Button { model.download() } label: { Text("Retry").frame(maxWidth: .infinity).padding(.vertical, 6) }
                        .buttonStyle(.borderedProminent).tint(ink).foregroundStyle(.white)
                }
            default:
                Button { model.download() } label: {
                    Text("Download the model (\(megabytes(ModelFiles.totalBytes)) MB)").frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent).tint(ink).foregroundStyle(.white)
            }
            Spacer()
            Text("huggingface.co/\(ModelFiles.repo)").font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 24).padding(.bottom, 24)
    }

    private var loadingContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            Spacer()
            ProgressView().tint(ink)
            Text("Preparing the model").font(.system(size: 27, weight: .medium, design: .serif))
            Text("The first load compiles the model for this iPhone; later loads use the saved result.")
                .font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 24)
    }

    private var indexingContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            Spacer(minLength: 12)
            Image(systemName: "books.vertical").font(.system(size: 44, weight: .light))
            Text("Making the example books searchable").font(.system(size: 31, weight: .medium, design: .serif))
            VStack(alignment: .leading, spacing: 10) {
                ForEach(model.library?.works ?? [], id: \.self) { work in
                    Text(work).font(.system(.body, design: .serif))
                }
            }
            VStack(alignment: .leading, spacing: 13) {
                HStack(alignment: .firstTextBaseline) {
                    Text(model.indexProgress.done.formatted()).font(.system(size: 42, weight: .medium, design: .rounded))
                    Text("/ \(model.bookCount.formatted()) sentences").font(.callout).foregroundStyle(.secondary)
                }
                .monospacedDigit().accessibilityIdentifier("index-progress")
                ProgressView(value: Double(model.indexProgress.done), total: Double(max(1, model.bookCount))).tint(ink)
                Text(String(format: "%.1f s elapsed  ·  %.1f ms / sentence",
                            model.indexProgress.elapsedSeconds, model.indexProgress.millisecondsPerSentence))
                    .font(.footnote.monospacedDigit()).foregroundStyle(.secondary)
                Text("This happens once.").font(.footnote)
            }
            Spacer()
            Text("Granite Embedding 97M · Core AI · on this iPhone").font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 24).padding(.bottom, 24)
    }

    private func failedContent(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Image(systemName: "exclamationmark.triangle").font(.largeTitle)
            Text("The model could not open").font(.title3.bold())
            Text(message).font(.callout).fixedSize(horizontal: false, vertical: true)
            Button("Try again") { model.retryLoad() }.buttonStyle(.borderedProminent).tint(ink).foregroundStyle(.white)
            Spacer()
        }
        .padding(24)
    }

    // MARK: Search

    private var searchContent: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "magnifyingglass").padding(.top, 3)
                    TextField("Describe what you are looking for…", text: $model.query, axis: .vertical)
                        .lineLimit(1...3)
                        .font(.system(size: 17))
                        .autocorrectionDisabled().textInputAutocapitalization(.never)
                        .focused($searchFocused)
                        .accessibilityIdentifier("search-query")
                    if !model.query.isEmpty {
                        Button { model.query = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                            .accessibilityLabel("Clear search")
                    }
                }
                .padding(14).background(.white, in: RoundedRectangle(cornerRadius: 14))
                Picker("Search in", selection: $model.filter) {
                    Text("All").tag(SearchModel.Filter.all)
                    Text("My notes (\(model.notes.count.formatted()))").tag(SearchModel.Filter.notes)
                }
                .pickerStyle(.segmented)
                HStack {
                    if let search = model.lastSearch {
                        Text("query \(String(format: "%.1f", search.embedAndRankMS)) ms · \(searchedCount.formatted()) sentences")
                    } else {
                        Text("\(model.bookCount.formatted()) book sentences · \(model.notes.count.formatted()) of yours")
                    }
                    Spacer(minLength: 4)
                }
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .accessibilityIdentifier("query-timing")
                if let notice = model.notice {
                    Text(notice).font(.caption.weight(.semibold))
                }
            }
            .padding(.horizontal, 20).padding(.bottom, 12)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if model.hits.isEmpty { emptyContent }
                    ForEach(model.hits) { hit in card(hit) }
                }
                .padding(.horizontal, 20).padding(.bottom, 24)
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    private var searchedCount: Int {
        model.filter == .all ? model.bookCount + model.notes.count : model.notes.count
    }

    @ViewBuilder private var emptyContent: some View {
        VStack(alignment: .leading, spacing: 14) {
            if model.filter == .notes && model.notes.isEmpty {
                Text("Nothing of yours yet.").font(.system(.title2, design: .serif))
                Text("Paste notes or import a .txt file; then search them by meaning.")
                    .font(.body).foregroundStyle(.secondary)
                Button("Add your text") { model.presentAddSheet() }.buttonStyle(.borderedProminent).tint(ink).foregroundStyle(.white)
            } else if model.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && model.filter == .notes {
                Text("Remember the meaning, not the words.").font(.system(.title2, design: .serif))
                Text("Search the \(model.notes.count.formatted()) sentences you added. All also searches two Sherlock Holmes books.")
                    .font(.body).foregroundStyle(.secondary)
            } else if model.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("Remember the meaning, not the words.").font(.system(.title2, design: .serif))
                Text("Search two Sherlock Holmes books, and any text you add with Add. Try one:")
                    .font(.body).foregroundStyle(.secondary)
                ForEach(model.presets, id: \.self) { preset in
                    Button { model.query = preset } label: {
                        Text(preset).font(.callout).multilineTextAlignment(.leading)
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .background(.white, in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                }
            } else if model.lastSearch != nil {
                Text("No match.").font(.system(.title3, design: .serif))
            }
        }
        .padding(.vertical, 22)
    }

    private struct Line: Identifiable {
        let id: String
        let text: String
    }

    private func card(_ hit: Hit) -> some View {
        let label: String, text: String, context: [Line], key: String
        if hit.source == .book, let library = model.library, library.sentences.indices.contains(hit.index) {
            let row = library.sentences[hit.index]
            (label, text, key) = ("\(row.work) · \(row.title)", row.text, row.id)
            context = library.context(at: hit.index).map { Line(id: $0.id, text: $0.text) }
        } else if hit.source == .note, model.notes.indices.contains(hit.index) {
            let row = model.notes[hit.index]
            (label, text, key) = ("My notes · \(row.source)", row.text, row.id)
            context = model.notes.context(at: hit.index).map { Line(id: $0.id, text: $0.text) }
        } else {
            (label, text, key, context) = ("", "", hit.id, [])
        }
        let isExpanded = expanded == key
        return Button {
            searchFocused = false
            withAnimation(.easeInOut(duration: 0.18)) { expanded = isExpanded ? nil : key }
        } label: {
            VStack(alignment: .leading, spacing: 9) {
                HStack(alignment: .firstTextBaseline) {
                    Text(label).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                    Spacer(minLength: 8)
                    Text(String(format: "%.2f", hit.score)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
                if isExpanded {
                    ForEach(context) { line in
                        Text(verbatim: line.text)
                            .font(.system(.body, design: .serif))
                            .foregroundStyle(line.id == key ? ink : ink.opacity(0.65))
                            .fontWeight(line.id == key ? .medium : .regular)
                            .multilineTextAlignment(.leading)
                    }
                } else {
                    Text(verbatim: text).font(.system(.body, design: .serif)).multilineTextAlignment(.leading)
                }
                if context.count > 1 {
                    HStack {
                        Text(isExpanded ? "Hide context" : "Read in context")
                        Spacer()
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    }
                    .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(hit.source == .note ? Color(red: 1, green: 0.98, blue: 0.9) : .white,
                        in: RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("hit-\(key)")
    }
}

struct AddTextSheet: View {
    @ObservedObject var model: SearchModel
    @State private var importing = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $model.draft.text)
                        .font(.system(size: 16))
                        .scrollContentBackground(.hidden)
                        .padding(6)
                        .disabled(model.draft.isAdding)
                        .accessibilityIdentifier("add-text")
                    if model.draft.text.isEmpty {
                        Text("Paste notes, an email, a recipe — anything you want to find again later.")
                            .foregroundStyle(.secondary).padding(.horizontal, 11).padding(.vertical, 14)
                            .allowsHitTesting(false)
                    }
                }
                .frame(minHeight: 240)
                .background(.white, in: RoundedRectangle(cornerRadius: 12))
                HStack {
                    Button { importing = true } label: { Label("Import .txt", systemImage: "doc.text") }
                        .disabled(model.draft.isAdding)
                    Spacer()
                    Text("From: \(model.draft.source)").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                if let message = model.draft.message {
                    Label(message, systemImage: "exclamationmark.circle").font(.callout.weight(.semibold))
                        .foregroundStyle(Color(red: 0.7, green: 0.15, blue: 0.1))
                        .accessibilityIdentifier("add-message")
                }
                if model.draft.isAdding {
                    VStack(alignment: .leading, spacing: 8) {
                        ProgressView(value: Double(model.draft.added), total: Double(max(model.draft.total, 1))).tint(ink)
                        HStack {
                            Text("Adding \(model.draft.added.formatted()) of \(model.draft.total.formatted()) sentences")
                                .font(.footnote.monospacedDigit())
                            Spacer()
                            Button("Cancel", role: .cancel) { model.cancelAdd() }
                        }
                    }
                }
                Text("The text is split into sentences; up to \(SearchModel.maxSentencesPerAdd.formatted()) at a time. It stays on this iPhone.")
                    .font(.footnote).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(20)
            .background(paper.ignoresSafeArea())
            .navigationTitle("Add your text")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { model.closeAddSheet() }.disabled(model.draft.isAdding)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") { model.addDraft() }.disabled(model.draft.isAdding)
                }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.plainText, .text]) { result in
                switch result {
                case .success(let url): model.importFile(url)
                case .failure(let error): model.draft.message = error.localizedDescription
                }
            }
        }
        .interactiveDismissDisabled(model.draft.isAdding)
        .foregroundStyle(ink)
        .preferredColorScheme(.light)
    }
}
