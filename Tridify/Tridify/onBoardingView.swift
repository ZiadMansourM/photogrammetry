//
//  onBoardingView.swift
//  Tridify
//
//  Created by Maged Alosali on 27/04/2023.
//

import SwiftUI

struct onBoardingView: View {
    var body: some View {
        NavigationView {
            GeometryReader { geo in
                VStack {
                    Spacer()
                    Image("icon")
                        .resizable()
                        .scaledToFit()
                        .frame(width: geo.size.width*0.4)
                        .frame(width: geo.size.width)
                    Text("Tridify")
                        .font(.system(size: geo.size.width/8))
                        .fontWeight(.light)
                        .kerning(15)

                    Text("Transform 2D images into 3D models.")
                        .font(.headline)
                        .foregroundColor(.black.opacity(0.65))
                        .fontWeight(.light)
                    
                    Spacer()
                    
                    NavigationLink {
                        // go to the sign page
                        Text("Create an account")
                    } label: {
                        HStack {
                            HStack {
                                Spacer()
                                Text ("Create Tridify Account")
                                    .font(.title2)
                                    .fontWeight(.medium)
                            
                                Spacer()
                                
                                Image(systemName: "arrowtriangle.forward.fill")
                                Spacer()
                            }
                            .padding()
                        }
                        .foregroundColor(.white)
                        .background(.black.opacity(0.9))
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .frame(width: geo.size.width*0.75)
                    }
                    
                    HStack {
                        Text ("Already have an account ?")
                        NavigationLink {
                            Text("Login page")
                        } label: {
                            Text("Log In")
                                .foregroundColor(.blue)
                        }
                    }
                    .font(.headline)
                    .fontWeight(.regular)
                    .padding(.vertical)
                    
                    Spacer()
                    
                    NavigationLink {
                        // go to the main view
                        Text("main view")
                    } label: {
                        HStack {
                            Text("Skip")
                            Image(systemName: "arrow.right")
                        }
                        .foregroundColor(.black.opacity(0.4))
                    }
                }
            }
        }
    }
}

struct onBoardingView_Previews: PreviewProvider {
    static var previews: some View {
        onBoardingView()
    }
}
