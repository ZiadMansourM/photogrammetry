//
//  HomeView.swift
//  Tridify
//
//  Created by Maged Alosali on 06/05/2023.
//

import SwiftUI

struct HomeView: View {
    
    @AppStorage("onBoarding") var onBoarding = false
    @State private var cameraViewOn = true
    @State private var isShowing = false
    var body: some View {
        GeometryReader { geo in
            if cameraViewOn {
                VStack(alignment: .trailing) {
                    Button {
                        isShowing.toggle()
                    } label: {
                        Image(systemName: "x.circle")
                            .font(.system(size: 40))
                            .fontWeight(.light)
                            .padding(.trailing, 8)
                            .padding(.top, 10)
                            .foregroundColor(Color(white: 0.85))
                    }
                    CustomCameraView()
                }
                .transition(.move(edge: .leading))
                .animation(.easeIn, value: cameraViewOn)
            }
            else {
                GalleryView(cameraViewOn: $cameraViewOn)
            }
        }
        .alert("Close Camera", isPresented: $isShowing) {
            HStack {
                Button("Close", role: .destructive){
                    cameraViewOn = false
                }
                Button("Cancel", role: .cancel){}
            }
        } message: {
            Text("Are you sure you want to close the camera?, All the captured images will be lost")
        }
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
        HomeView().preferredColorScheme(.dark)
    }
}
