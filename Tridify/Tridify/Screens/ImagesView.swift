//
//  ImagesView.swift
//  Tridify
//
//  Created by Maged Alosali on 15/05/2023.
//

import SwiftUI

struct ImagesView: View {
    @State private var smallImageOffset: CGFloat = 0
    @State private var currentIndex: Int = 0
    @Binding private var capturedData:[Data]
    @State private var isShowingAlert = false
    @Binding private var deleteLast: Bool
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        GeometryReader { geo in
            VStack (alignment: .leading) {

                Text("Count: \(capturedData.count)")
                    .font(.headline)
                    .fontWeight(.light)
                    .kerning(2)
                    .padding(20)
                if capturedData.isEmpty {
                    Color.clear
                        .frame(width: geo.size.width, height: geo.size.height * 0.75)
                }
                else {
                    Image(uiImage: UIImage(data: capturedData[currentIndex])!)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height * 0.75)
                        .onAppear {
                            currentIndex = 0
                        }
                }
                Spacer()
                HStack {
                    ScrollView(.horizontal, showsIndicators: false){
                        HStack {
                            ForEach(capturedData.indices, id: \.self){ i in
                                Button {
                                    currentIndex = i
                                } label: {
                                    ZStack {
                                        Image(uiImage: UIImage(data: capturedData[i])!)
                                            .resizable()
                                            .scaledToFill()
                                            .clipShape(RoundedRectangle(cornerRadius: 5))
                                            .frame(width: geo.size.width*0.125, height: geo.size.width*0.11)
                                        Text("\(i + 1)")
                                            .font(.caption)
                                            .bold()
                                            .foregroundColor(.black)
                                            .background(
                                                Circle()
                                                    .foregroundColor(.white)
                                                    .background(.clear)
                                                    .frame(width:50)
                                            )
                                    }
                                }
                            }
                        }
                        .padding(.trailing, geo.size.width * 0.78)
                        .overlay {
                            GeometryReader { smallGeo in
                                Color.clear
                                    .preference(key: ScrollOffsetPreferenceKey.self, value: smallGeo.frame(in: .named("scrollView")).origin.x)
                            }
                            .onPreferenceChange(ScrollOffsetPreferenceKey.self){ value in
                                smallImageOffset = value
                                
                                currentIndex = calculateIndex(sizeOfImage: geo.size.width*0.125 + 8, offsetValue: smallImageOffset)
                                
                            }
                        }
                    }
                    Button {
                        isShowingAlert.toggle()
                    } label: {
                        Image(systemName: "trash")
                            .font(.title2)
                            .foregroundColor(.link)
                            .padding(.horizontal)
                    }
                }
                Spacer()
            }
        }
        .navigationTitle("Images review")
        .alert("Delete Image", isPresented: $isShowingAlert) {
            HStack {
                Button("Delete", role: .destructive){
                    let indexToDelete = currentIndex
                    
                    if capturedData.count == 1 {
                        deleteLast = true
                        dismiss()
                    }
                    
                    if indexToDelete == capturedData.count - 1 {
                        currentIndex -= 1
                    }
                    capturedData.remove(at: indexToDelete)
                    
                }
                Button("Cancel", role: .cancel) {}
            }
        } message: {
            VStack {
                Text("Are you sure you want to delete this image number \(currentIndex + 1)?")
            }
        }

    }
    
    init(capturedData:  Binding<[Data]>, deleteLast: Binding<Bool>) {
        _capturedData = capturedData
        _deleteLast = deleteLast
    }
    
    private func calculateIndex(sizeOfImage: CGFloat, offsetValue: CGFloat) -> Int {
        var index = 0
        index = Int(-offsetValue/sizeOfImage)
        if index < 0 {
            index = 0
        }
        index = min(index, capturedData.count - 1)
        return index
    }
}


struct ScrollOffsetPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}


